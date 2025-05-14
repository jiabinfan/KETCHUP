import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from transformers import GenerationConfig

def shift_right(tensor, fill_token_id):
    assert len(tensor.shape) == 2
    start_ids = torch.ones((tensor.shape[0], 1)).to(
        tensor) * fill_token_id
    return torch.cat([start_ids, tensor[..., :-1]], dim=-1)
def shift_left(tensor, fill_token_id):
    assert len(tensor.shape) == 2
    start_ids = torch.ones((tensor.shape[0], 1)).to(
        tensor) * fill_token_id
    return torch.cat([tensor[..., 1: ],start_ids], dim=-1)
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    # rougeLSum expects newline after each sentence
    # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    # labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels
def kl_sum(kl_loss, mask):
    if mask is not None:
        mask = mask.unsqueeze(-1)
        kl_loss = kl_loss * mask
        kl_loss = kl_loss.sum() / mask.shape[0]
    else:
        kl_loss = kl_loss.mean()  # Average over all elements if no mask is provided
    return kl_loss
def KLD(student_logits, teacher_logits, mask=None):
    student_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    kl_loss = F.kl_div(student_probs, teacher_probs, reduction='none')
    kl_loss = kl_sum(kl_loss, mask)
    return kl_loss

def JS(s_id_s_logits, s_id_t_logits, t_id_s_logits, t_id_t_logits, s_ids, t_ids, pad_token_id):
    dec_mask_s_ids = s_ids.ne(pad_token_id)
    dec_mask_t_ids = t_ids.ne(pad_token_id)
    loss_js_s_ids = calc_js_loss(dec_mask_s_ids, s_id_s_logits, s_id_t_logits, 
        s_wght=0.5, t_wght=0.5, add_eps=True)

    loss_js_t_ids = calc_js_loss(dec_mask_t_ids, t_id_s_logits, t_id_t_logits, 
        s_wght=0.5, t_wght=0.5, add_eps=True)    
    return 0.5*loss_js_s_ids + 0.5*loss_js_t_ids

def TVD(s_id_s_logits, s_id_t_logits, t_id_s_logits, t_id_t_logits, s_ids, t_ids, pad_token_id):
 
    dec_mask_s_ids = s_ids.ne(pad_token_id)
    dec_mask_t_ids = t_ids.ne(pad_token_id)

    loss_tvd_s_ids = calc_tvd_loss(dec_mask_s_ids, s_id_s_logits, s_id_t_logits)
    loss_tvd_t_ids = calc_tvd_loss(dec_mask_t_ids, t_id_s_logits, t_id_t_logits) 
 
    return 0.5*loss_tvd_s_ids + 0.5*loss_tvd_t_ids

def calc_tvd_loss(mask, s_logits, t_logits):
    s_logits = F.softmax(s_logits, dim=-1)
    t_logits = F.softmax(t_logits, dim=-1)
    sel_mask = mask[:, :, None].expand_as(s_logits)
    vocab_size = s_logits.size(-1)
    s_logits_slct = torch.masked_select(s_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
    t_logits_slct = torch.masked_select(t_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
    s_logits_slct = s_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
    t_logits_slct = t_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
    assert t_logits_slct.size() == s_logits_slct.size()
    loss_tvd = (0.5 * torch.abs(s_logits_slct-t_logits_slct)).sum(dim=-1).mean()
    return loss_tvd

def calc_js_loss(mask, s_logits, t_logits, s_wght=0.5, t_wght=0.5, add_eps=False):
    # mask has False at padding_idx
    sel_mask = mask[:, :, None].expand_as(s_logits)
    vocab_size = s_logits.size(-1)
    s_logits_slct = torch.masked_select(s_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
    t_logits_slct = torch.masked_select(t_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
    s_logits_slct = s_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
    t_logits_slct = t_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
    assert t_logits_slct.size() == s_logits_slct.size()
    q_prob = t_wght*F.softmax(t_logits_slct , dim=-1) \
            + s_wght*F.softmax(s_logits_slct , dim=-1)
    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    loss_ce =criterion(
            torch.log(q_prob+1e-10), # bottom
            F.softmax(s_logits_slct , dim=-1) + 1e-40# up
        )* 1.

    
    # loss_ce = (
    #     nn.KLDivLoss(
    #         torch.log(q_prob+1e-10), # bottom
    #         F.softmax(s_logits_slct , dim=-1)# up
    #     )
    #     * 1.)
    return loss_ce

class TemperatureCrossEntropy(torch.nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        if 'temperature' in kwargs:
            self.temperature = kwargs.pop('temperature')
        else:
            self.temperature = 1.0
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        return super().forward(input * self.temperature, target)
    
class Net(nn.Module):
    def __init__(self, seq2seq_model, args, teacher=None, max_length=48,logits_processor=None, tokenizer=None, init_model=None, gen_kwargs=None):
        super(Net, self).__init__()
        self.seq2seq_model = seq2seq_model
        #fixed teacher
        self.teacher = teacher
        self.init_model = init_model
        if teacher:
            for param in self.teacher.parameters():
                param.requires_grad = False
        if init_model:
            for param in self.init_model.parameters():
                param.requires_grad = False
        # criterion_params={
        #     'ignore_index': self.seq2seq_model.config.pad_token_id,
        #     'temperature': 1.0,
        #     'label_smoothing': 0.0,
        #     'reduction': 'none',}
        # self.loss_ce = TemperatureCrossEntropy(**criterion_params)
        self.loss_ce=nn.CrossEntropyLoss(ignore_index=self.seq2seq_model.config.pad_token_id)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.max_length = max_length
        self.args = args
        self.logits_proc = logits_processor
        self.tokenizer = tokenizer
        self.gen_kwargs = gen_kwargs
    
    def reinforce(self, input_ids,output_ids, config, returns, denom=1.0, reward_clip=None, 
                  entropy_coef=0.0, method='None', use_kl=None, kl_coef=0.001):
        dec_input = shift_right(output_ids, config.decoder_start_token_id)
        attention_mask = (output_ids != config.pad_token_id)
                
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': input_ids != config.pad_token_id,
            'decoder_input_ids': dec_input,
            'decoder_attention_mask': attention_mask,
            'return_dict': True,
            'use_cache': False,
        }

        # Forward pass through the model
        output = self.seq2seq_model(**model_inputs)
        logits = output.logits  # (BATCH, LENGTH, VOCAB)

        probs = torch.softmax(logits, dim=-1).masked_fill(attention_mask.logical_not()[..., None], 0.0)

        with torch.no_grad():
            returns = returns / denom
            rho = torch.gather(probs, -1, output_ids[..., None]).squeeze(-1)
            rho = rho.masked_fill(attention_mask.logical_not(), 0.0)
            rho_returns = returns * rho
            if reward_clip is not None:
                rho_returns = torch.clamp(rho_returns, min=-reward_clip, max=reward_clip)
            rho_returns = rho_returns.masked_fill(attention_mask.logical_not(), 0.0)

        loss = 0
        lprobs = torch.log_softmax(logits, dim=-1)

        if entropy_coef != 0:
            entropy = torch.mean(-lprobs, -1)
            entropy = entropy.masked_fill(attention_mask.logical_not(), 0.0)
            entropy = entropy.sum() / attention_mask.sum()
            loss = loss + entropy * entropy_coef

        lprobs = torch.gather(lprobs, -1, output_ids[..., None]).squeeze(-1)  # (B, L)
        lprobs = lprobs.masked_fill(attention_mask.logical_not(), 0.0)

        if method == 'mean':
            rho_returns = rho_returns-rho_returns.mean().detach()
        elif method =='min-variance':
            # Assuming lprobs requires gradients
            lprobs.retain_grad()  # Ensure gradients are retained for lprobs

            # Compute the gradients
            lprobs.backward(torch.ones_like(lprobs), retain_graph=True)

            # Use the gradients for further calculations
            grads = lprobs.grad

            a = rho_returns * torch.square(grads)
            b = torch.square(grads)
            min_variance_baseline =a.mean()/b.mean()
            rho_returns = rho_returns - min_variance_baseline.detach()

        loss = loss + (-lprobs * rho_returns).sum() / attention_mask.sum()

        return loss

    def forward(self, input_ids, attention_mask, labels, tokenizer=None,train_type="distill_tvd"):
        num_violation = 0
        num_neg_loss = 0
        generated_ids = input_ids  # Start with the input_ids

        if train_type == "distill_seqkd":
            with torch.no_grad():  
                t_ids = self.teacher.generate(input_ids, 
                                              attention_mask=attention_mask,
                                              length_penalty=1.0, 
                                              max_length=self.max_length, 
                                              num_beams=1, 
                                              num_return_sequences=1,
                                              logits_processor=self.logits_proc,
                                              return_dict_in_generate=False,
                                              do_sample=False)
                
                t_ids = shift_left(t_ids,tokenizer.pad_token_id)
                t_ids_target = t_ids.view(-1)
                
            t_ids_s_output = self.seq2seq_model(input_ids=input_ids, attention_mask=attention_mask, labels=t_ids,
                                                logits_processor=self.logits_proc,
                                                return_dict_in_generate=False,)
            t_ids_s_logits = t_ids_s_output.logits
            real_stu_logits = t_ids_s_logits.view(-1, t_ids_s_logits.size(-1))
            ce_loss = self.loss_ce(real_stu_logits, t_ids_target)
      
            return ce_loss
        elif train_type == "distill_kl":
            with torch.no_grad():  
                t_ids = self.teacher.generate(input_ids, max_length=self.max_length, num_beams=1, do_sample=False)
                # t_ids: [batch_size, max_length]
                t_ids = shift_left(t_ids,tokenizer.pad_token_id)
                t_ids_t_output = self.teacher(input_ids=input_ids, attention_mask=attention_mask, labels=t_ids)
                t_ids_t_logits = t_ids_t_output.logits

            t_ids_s_output = self.seq2seq_model(input_ids=input_ids, attention_mask=attention_mask, labels=t_ids)
            t_ids_s_logits = t_ids_s_output.logits
            #print("t_ids", t_ids[0])
            decoder_attention_mask = (t_ids != tokenizer.pad_token_id)
            kl_loss = KLD(t_ids_s_logits, t_ids_t_logits, decoder_attention_mask)
   
            return  kl_loss

        elif train_type == "distill_js":
            s_ids = self.seq2seq_model.generate(input_ids, max_length=self.max_length, num_beams=1, do_sample=False)
            with torch.no_grad():  
                t_ids = self.teacher.generate(input_ids, max_length=self.max_length, num_beams=1, do_sample=False)
                # t_ids: [batch_size, max_length]
                t_ids = shift_left(t_ids,tokenizer.pad_token_id)
                t_ids_t_output = self.teacher(input_ids=input_ids, attention_mask=attention_mask, labels=t_ids)
                t_ids_t_logits = t_ids_t_output.logits

                s_ids_t_output = self.teacher(input_ids=input_ids, attention_mask=attention_mask, labels=s_ids)
                s_ids_t_logits = s_ids_t_output.logits               
                
            t_ids_s_output = self.seq2seq_model(input_ids=input_ids, attention_mask=attention_mask, labels=t_ids)
            t_ids_s_logits = t_ids_s_output.logits
            s_ids_s_output = self.seq2seq_model(input_ids=input_ids, attention_mask=attention_mask, labels=s_ids)
            s_ids_s_logits = s_ids_s_output.logits
            loss = JS(s_ids_s_logits, s_ids_t_logits, t_ids_s_logits, t_ids_t_logits, s_ids, t_ids, tokenizer.pad_token_id)

            return loss

        elif train_type == "distill_tvd":
            s_ids = self.seq2seq_model.generate(input_ids, max_length=self.max_length, num_beams=1, do_sample=False,
                                                logits_processor=self.logits_proc,
                                                return_dict_in_generate=False,)
            with torch.no_grad():  
                t_ids = self.teacher.generate(input_ids, max_length=self.max_length, num_beams=1, do_sample=False,
                                              logits_processor=self.logits_proc,
                                              return_dict_in_generate=False,)
                # t_ids: [batch_size, max_length]
                t_ids = shift_left(t_ids,tokenizer.pad_token_id)
                t_ids_t_output = self.teacher(input_ids=input_ids, attention_mask=attention_mask, labels=t_ids)
                t_ids_t_logits = t_ids_t_output.logits

                s_ids_t_output = self.teacher(input_ids=input_ids, attention_mask=attention_mask, labels=s_ids)
                s_ids_t_logits = s_ids_t_output.logits               
                
            t_ids_s_output = self.seq2seq_model(input_ids=input_ids, attention_mask=attention_mask, labels=t_ids)
            t_ids_s_logits = t_ids_s_output.logits
            s_ids_s_output = self.seq2seq_model(input_ids=input_ids, attention_mask=attention_mask, labels=s_ids)
            s_ids_s_logits = s_ids_s_output.logits
            loss = TVD(s_ids_s_logits, s_ids_t_logits, t_ids_s_logits, t_ids_t_logits, s_ids, t_ids, tokenizer.pad_token_id)

            return loss

        elif train_type == "sft":
            # Forward pass for seq2seq_model

            real_stu_output = self.seq2seq_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss = real_stu_output.loss
               
        return total_loss

    def generate(self, input_ids, attention_mask, do_sample=False, top_k=1, max_new_tokens=64):
        model=self.seq2seq_model
        gen_cfg = GenerationConfig(
            max_new_tokens = max_new_tokens,   # ← generate up to 128 new tokens
            do_sample      = False,
            pad_token_id   = self.tokenizer.pad_token_id,
            eos_token_id   = self.tokenizer.eos_token_id,
        )
        generated_tokens = model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    generation_config=gen_cfg,
                )
        prompt_lens = input_ids.shape[1]
        return generated_tokens
    
    def math_generate(self, input_ids, attention_mask, do_sample = False, top_k = 1,**gen_kwargs):
        model=self.seq2seq_model
        generated_tokens = model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    logits_processor=self.logits_proc,
                    return_dict_in_generate=False,
                    
                    **gen_kwargs, 
                )
        return generated_tokens
        