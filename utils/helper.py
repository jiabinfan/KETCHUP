import torch
import re
from transformers import LogitsProcessor

def tool_fn(expr):
    raw_result = eval(expr)
    f_res = float(raw_result)
    if abs(int(f_res)-f_res) < 0.0001:
        return str(int(f_res))
    else:
        return str(round(f_res, 2))


class CustomLogitsProcessor(LogitsProcessor):

    def __init__(self, tokenizer, tool):
        self.tokenizer = tokenizer
        self.tool = tool
        self.eq_token_id = tokenizer.convert_tokens_to_ids('▁=')
        self.right_token_id = tokenizer.convert_tokens_to_ids(']')
        self.left_token_id = tokenizer.convert_tokens_to_ids('▁[')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        bz = scores.size(0)
        for b in range(bz):
            if input_ids[b][-1].item() == self.tokenizer.pad_token_id:
                continue
            if input_ids[b][-1].item() == self.tokenizer.eos_token_id:
                continue
            cur_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[b])
            meet_right = False
            meet_eq = None
            for i in range(len(cur_tokens)-1, -1, -1):
                if cur_tokens[i] == ']':
                    break
                elif cur_tokens[i] == '▁=':
                    meet_eq = i
                elif cur_tokens[i] == '▁[' and meet_eq is not None:
                    string = ''.join(cur_tokens[i+1:meet_eq])
                    string = re.sub('▁', ' ', string)
                    try:
                        res = self.tool(string)
                    except:
                        break
                    target_token_ids = self.tokenizer.encode(res, add_special_tokens=False, return_tensors='pt')[0]
                    if all(torch.eq(target_token_ids, input_ids[b][-len(target_token_ids):].cpu())):
                        scores[b, self.right_token_id] = 10e7
                    elif input_ids[b][-1] == self.eq_token_id:
                        scores[b, target_token_ids[0]] = 10e7
                    else:
                        gen_token_ids = input_ids[b][meet_eq+1:]
                        l = len(gen_token_ids)
                        scores[b, target_token_ids[l]] = 10e7
                    break
        return scores

def greedy_search(model, enc_input, device, args, config):
    model_kwargs = dict()
    model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
        enc_input, model_kwargs)

    if "decoder_input_ids" in model_kwargs:
        input_ids = model_kwargs.pop("decoder_input_ids")
    else:
        input_ids = model._prepare_decoder_input_ids_for_generation(
            enc_input.shape[0]
        )
    input_ids, model_kwargs = model._expand_inputs_for_generation(
        input_ids,
        is_encoder_decoder=config.is_encoder_decoder, **model_kwargs
    )
    ready_mask = torch.zeros((enc_input.shape[0], 1)).bool().to(device=device)
    
    max_length = max(
        args.min_max_length,
        min(
            args.max_length,
            int(args.length_ratio * args.max_tokens - input_ids.numel()) // input_ids.shape[0]
        )
    )
    b_probs = []
    for l in range(max_length):
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)
        outputs = model(**model_inputs,
                        return_dict=True,
                        output_hidden_states=config.output_hidden_states,
                        output_attentions=config.output_attentions)
        
        # if l==max_length-1:
        #     teacher_logits = outputs.logits / args.temperature
        
        logits = outputs.logits[:, -1, :] / args.temperature 

        def top_k_mask_out(t, k, flat=False):
            _list = torch.topk(t, k, -1)[0] # [B,1]
            lb = _list[..., -1, None] # [B,1]

            if flat:
                return torch.where(t >= lb, 0.0, -torch.inf)
            return torch.masked_fill(t, t < lb, -torch.inf)
        if args.topk is not None:
            logits = top_k_mask_out(logits, args.topk, flat=(l==0) and args.init_flat)

        if args.mask_extra:
            logits[..., 32000:] = -torch.inf

        prob = torch.softmax(logits, -1)

        try:
            next_tokens = torch.multinomial(prob, 1) # [B,1]
        except RuntimeError:
            next_tokens = torch.argmax(prob, -1, keepdim=True)
        b_probs.append(prob.gather(-1, next_tokens))

        next_tokens.masked_fill_(ready_mask, config.pad_token_id)
        eos_mask = next_tokens == config.eos_token_id
        
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=config.is_encoder_decoder
        )

        ready_mask = eos_mask | ready_mask 
        if ready_mask.all():
            break
    
    b_probs = torch.cat(b_probs, dim=-1)
    input_ids = input_ids[..., 1:]

    final_logits = outputs.logits / args.temperature

    # if l==max_length-1:
    #     return input_ids, b_probs, ready_mask, teacher_logits
    # else:

    return input_ids, b_probs, ready_mask, final_logits

def beam_search(model, enc_input, device, args, config, beam_size=None):
    model_kwargs = dict()
    model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
        enc_input, model_kwargs)

    if "decoder_input_ids" in model_kwargs:
        dec_input_ids = model_kwargs.pop("decoder_input_ids")
    else:
        dec_input_ids = model._prepare_decoder_input_ids_for_generation(enc_input.shape[0])


    input_ids, model_kwargs = model._expand_inputs_for_generation(
        dec_input_ids,
        expand_size=beam_size,
        is_encoder_decoder=config.is_encoder_decoder, 
        **model_kwargs
    )
    ready_mask = torch.zeros((enc_input.shape[0]* beam_size, 1)).bool().to(device=device)

    max_length = max(
        args.min_max_length,
        min(
            args.max_length,
            int(args.length_ratio * args.max_tokens - dec_input_ids.numel()) // dec_input_ids.shape[0]
        )
    )
    
    beam_scores = torch.zeros((enc_input.shape[0], beam_size), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((-1))

    for l in range(max_length):
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)
        outputs = model(**model_inputs,
                        return_dict=True,
                        output_hidden_states=config.output_hidden_states,
                        output_attentions=config.output_attentions)
        
        logits = outputs.logits[:, -1, :] / args.temperature 
        

        def top_k_mask_out(t, k, flat=False):
            _list = torch.topk(t, k, -1)[0]
            lb = _list[..., -1, None]

            if flat:
                return torch.where(t >= lb, 0.0, -torch.inf)
            return torch.masked_fill(t, t < lb, -torch.inf)
        
        if args.topk is not None:
            logits = top_k_mask_out(logits, beam_size, flat=(l==0) and args.init_flat)

        if args.mask_extra:
            logits[..., 32000:] = -torch.inf

        scores = torch.log_softmax(logits, -1)
        scores = scores + beam_scores[:, None].expand_as(scores) # (batch_size * num_beams, vocab_size)

        # reshape for beam search
        vocab_size = scores.shape[-1]
        total_scores = scores.view(enc_input.shape[0], beam_size * vocab_size)
        next_scores, next_tokens = torch.topk(total_scores, beam_size, dim=1, largest=True, sorted=True)

        beam_indices = next_tokens // vocab_size
        next_tokens = next_tokens % vocab_size

        # if ready, the just use padding
        next_tokens=next_tokens.view(-1, 1).masked_fill_(ready_mask, config.pad_token_id)

        # Update input_ids and b_probs with the selected tokens and probabilities
        input_ids = input_ids.view(enc_input.shape[0], beam_size, -1)
        cur_input_ids = torch.gather(input_ids, 1, beam_indices.unsqueeze(-1).expand_as(input_ids)).view(enc_input.shape[0] * beam_size, -1)
        
        input_ids = torch.cat([cur_input_ids, next_tokens], dim=-1)

        beam_scores = next_scores.view(-1)
        # b_probs.append(torch.gather(torch.softmax(logits, -1), 1, next_tokens))

        ready_mask = ready_mask.view(enc_input.shape[0], beam_size)
        eos_mask = next_tokens == config.eos_token_id
        ready_mask = torch.gather(ready_mask, 1, beam_indices).view(-1, 1) | eos_mask # (batch_size * num_beams, 1)
        
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=config.is_encoder_decoder
        )

        if ready_mask.all():
            break

    input_ids = input_ids[..., 1:]
    b_probs = torch.ones(input_ids.shape[0], input_ids.shape[1]).to(device)

    return input_ids, b_probs, ready_mask

def shift_left(tensor, fill_token_id):
    assert len(tensor.shape) == 2
    start_ids = torch.ones((tensor.shape[0], 1)).to(
        tensor) * fill_token_id
    return torch.cat([tensor[..., 1:]], start_ids, dim=-1)

def shift_right(tensor, fill_token_id):
    assert len(tensor.shape) == 2
    start_ids = torch.ones((tensor.shape[0], 1)).to(
        tensor) * fill_token_id
    return torch.cat([start_ids, tensor[..., :-1]], dim=-1)

def calc_returns(rewards):
    flip_rewards = rewards.flip([-1])
    cumsum_rewards = torch.cumsum(flip_rewards, dim=-1)
    returns = cumsum_rewards.flip([-1])

    return returns

def n_step_reward(reward, selection_value, next_state_value, n_step=1):
    n=selection_value.shape[1]
    for t in range(0,n):
        
        if t+n_step < n and t%n_step!=0: 
            reward[:, t] = 0
        if t+n_step >= n:
            if (n-1)%n_step==0 and t%n_step!=0:
                    reward[:, t] = 0
            elif (n-1)%n_step!=0:
                if t%n_step==0:
                    reward[:, t] = selection_value[:, t] - next_state_value[:, -((1+n_step)%n)]
                elif t==n-1:pass # no change for the last one
                else: 
                    reward[:, t] = 0
    
    return reward

def n_step_reward_shift(reward, selection_value, next_state_value, n_step=1):

    # reward=n_step_reward(reward, selection_value, next_state_value, n_step=1)
    returns=torch.zeros_like(reward)
    for i in range(0, reward.shape[1], n_step):
        returns[:, i]=reward[:,i:reward.shape[1]].sum(dim=-1)
    
    return returns

def n_step_reward_shift_upgrade(reward, selection_value, next_state_value, n_step=1):
    n=selection_value.shape[1]
    for t in range(0,n):
        if t+n_step >= n-1:
            if t!=n-1:
                if n_step < next_state_value.shape[1]:
                    reward[:, t] = selection_value[:, t] - next_state_value[:, -((1+n_step)%n)]
            elif t==n-1:
                pass # no change for the last one
    
    returns=torch.zeros_like(reward)
    for k in range(n_step):
        mask=torch.zeros_like(reward)
        mask[:,k::n_step]=1
        mask[:,n-1]=1
        shifted_rewards=reward*mask
        _return=calc_returns(shifted_rewards)
        returns[:,k::n_step]=_return[:,k::n_step]
    
    return returns

import os
from collections import deque
import shutil

class CheckpointManager:
    def __init__(self, save_dir, max_ckpts=5):
        self.save_dir = save_dir
        self.max_ckpts = max_ckpts
        self.best_ckpts = deque()  # Store (bleu2, path) tuples
        self.min_score = 0

    def add_ckpt_inqueue(self, cur_score, path):
        
        self.best_ckpts.append((cur_score, path))
        self.best_ckpts = deque(
            sorted(self.best_ckpts, key=lambda x: x[0], reverse=True)
        )
        self.min_score = self.best_ckpts[-1][0] if self.best_ckpts else 0

    def remove_worst_ckpt(self):
        if len(self.best_ckpts) > self.max_ckpts:
            _, dir_path = self.best_ckpts.pop()
            self.min_score = self.best_ckpts[-1][0] if self.best_ckpts else 0
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f'Removed checkpoint in {dir_path}')
            else:
                print(f'Checkpoint in {dir_path} does not exist')

    def should_save_ckpt(self, cur_score):
        # priority: higher cur_score
        if cur_score > self.min_score:
            print("Improved...")
            return True
        if len(self.best_ckpts) < self.max_ckpts:
            return True
        else:
            return False





