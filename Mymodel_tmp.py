import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from transformers import GenerationConfig

# causal-LM friendly helpers
def mask_prompt(labels, prompt_lens, pad_val=-100):
    """Mask prompt tokens so they don't contribute to loss."""
    for i, l in enumerate(prompt_lens):
        labels[i, :l] = pad_val
    return labels

def slice_generation(all_ids, prompt_lens):
    """Keep only newly generated tokens."""
    return torch.stack([ids[l:] for ids, l in zip(all_ids, prompt_lens)])

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
    return loss_ce
def concat_with_eos(a: torch.Tensor,
                    b: torch.Tensor,
                    eos_id: int) -> torch.Tensor:
    """
    Concatenate two sequences with a single EOS token in-between.
    Shape: (B, L₁) + (B,1) + (B, L₂)  ->  (B, L₁+L₂+1)
    """
    B = a.size(0)
    eos = torch.full((B, 1), eos_id, dtype=a.dtype, device=a.device)
    return torch.cat([a, eos, b], dim=1)
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig)

class Net(nn.Module):
    def __init__(self, student=None, args=None, teacher=None, max_length=48,logits_processor=None, tokenizer=None, init_model=None, gen_kwargs=None):
        super(Net, self).__init__()

        # --- student --------------------------------------------------------
        self.student = student
        self.tokenizer = tokenizer
        self.logits_processor = logits_processor
        # guarantee pad-id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # ! :contentReference[oaicite:3]{index=3}
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id 
        # --- fixed teacher (optional) --------------------------------------
        self.teacher = teacher
        if teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False

        # loss objects
        self.ce_loss  = nn.CrossEntropyLoss(ignore_index=-100)
        self.kl_scale = 1
        self.max_new  = max_length
        self.gcfg = GenerationConfig(
            pad_token_id = self.pad_id,
            eos_token_id = self.tokenizer.eos_token_id,
            do_sample      = False,
            num_beams=1,
            max_new_tokens = self.max_new
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        labels=None,
        train_type="distill_tvd",
        prompt_lens=None,
    ):
        prompt_ids = input_ids
        B = prompt_ids.size(0)
        if prompt_lens is None:
            prompt_lens = [prompt_ids.size(1)] * B
        attention_mask = prompt_ids.ne(self.pad_id)


        # ----------------- generate teacher / student continuations ------
        with torch.no_grad():
            t_full = self.teacher.generate(
                prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new,
                generation_config=self.gcfg,
            )
        t_ids = slice_generation(t_full, prompt_lens)
        #print("9999999999", prompt_ids[0], t_ids[0])
        s_full = self.student.generate(
            prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new,
            generation_config=self.gcfg,
        )
        # print(111111111, prompt_ids)
        # print(222222222, self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True))
        s_ids = slice_generation(s_full, prompt_lens)

        # convenience cat
        def _concat(pr, cont):
            return torch.cat([pr, cont], dim=1)

        # ------------------------------------------------------------------
        # Distill – KL (prompt + teacher ids)
        # ------------------------------------------------------------------
        if train_type == "distill_kl":
            seq = concat_with_eos(prompt_ids, t_ids, self.eos_id) 
            # print(555555, self.tokenizer.batch_decode(t_full, skip_special_tokens=False))
            # print(666666, self.tokenizer.batch_decode(seq, skip_special_tokens=False))
            attn = seq.ne(self.pad_id)
            with torch.no_grad():
                t_out = self.teacher(seq, attention_mask=attn)
            s_out = self.student(seq, attention_mask=attn)
            return KLD(s_out.logits, t_out.logits, mask=attn)

        # ------------------------------------------------------------------
        # Distill – JS / TVD (four-view pattern)
        # ------------------------------------------------------------------
        if train_type in { "distill_tvd"}:
            loss_kind = "tvd"
            return self._js_tvd_loss(
                prompt_ids, prompt_lens, s_ids, t_ids, loss_kind
            )

        raise ValueError(train_type)

    # ===================================================================== #
    #  JS / TVD helper                                                      #
    # ===================================================================== #
    def _js_tvd_loss(
        self, prompt_ids, prompt_lens, s_ids, t_ids, mode="js"
    ):
        s_seq = torch.cat([prompt_ids, s_ids], 1)
        t_seq = torch.cat([prompt_ids, t_ids], 1)

        s_mask = s_seq.ne(self.pad_id)
        t_mask = t_seq.ne(self.pad_id)

        # student on student ids
        s_s = self.student(s_seq, attention_mask=s_mask).logits
        # teacher on student ids
        with torch.no_grad():
            s_t = self.teacher(s_seq, attention_mask=s_mask).logits
        # student on teacher ids
        t_s = self.student(t_seq, attention_mask=t_mask).logits
        # teacher on teacher ids
        with torch.no_grad():
            t_t = self.teacher(t_seq, attention_mask=t_mask).logits

        return TVD(s_s, s_t, t_s, t_t, s_seq, t_seq, self.pad_id)

    # ===================================================================== #
    #  REINFORCE                                                            #
    # ===================================================================== #
    def reinforce(
        self,
        input_ids: torch.Tensor,
        returns: torch.Tensor,
        entropy_coef=0.0,
        baseline="mean",
        reward_clip=None,
    ):
        prompts = input_ids
        B, P = prompts.shape
        attn = prompts.ne(self.pad_id)

        # sample continuations
        full = self.student.generate(
            prompts,
            attention_mask=attn,
            max_new_tokens=self.max_new,
            generation_config=self.gcfg,
        )
        samples = slice_generation(full, [P] * B)
        seq = torch.cat([prompts, samples], 1)
        attn_seq = seq.ne(self.pad_id)

        logits = self.student(seq, attention_mask=attn_seq).logits
        logp = F.log_softmax(logits, -1)
        act_logp = torch.gather(
            logp[:, P - 1 : -1, :], -1, samples.unsqueeze(-1)
        ).squeeze(-1)

        if baseline == "mean":
            returns = returns - returns.mean()
        elif baseline == "min-variance":
            act_logp.retain_grad()
            act_logp.backward(torch.ones_like(act_logp), retain_graph=True)
            g = act_logp.grad
            returns = returns - (returns * g.square()).mean() / g.square().mean()

        if reward_clip:
            returns = torch.clamp(returns, -reward_clip, reward_clip)

        loss = -(act_logp.mean(1) * returns).mean()
        if entropy_coef:
            ent = -(logp.exp() * logp).sum(-1).mean()
            loss -= entropy_coef * ent
        return loss

    def generate(self, input_ids, attention_mask, do_sample=False, top_k=1, max_new_tokens=64):
        gen_cfg = GenerationConfig(
            max_new_tokens = max_new_tokens,   # ← generate up to 128 new tokens
            do_sample      = False,
            num_beams = 1,
            pad_token_id   = self.tokenizer.pad_token_id,
            eos_token_id   = self.tokenizer.eos_token_id,
        )
        generated_tokens = self.student.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    generation_config=gen_cfg,
                )
        return generated_tokens
    def math_generate(self, input_ids, attention_mask, do_sample = False, top_k = 1,max_new_tokens=256, **gen_kwargs):
        gen_cfg = GenerationConfig(
            max_new_tokens = max_new_tokens,   # ← generate up to 128 new tokens
            do_sample      = False,
            pad_token_id   = self.tokenizer.pad_token_id,
            eos_token_id   = self.tokenizer.eos_token_id,
        )
        
        generated_tokens = self.student.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    logits_processor=self.logits_processor,
                    return_dict_in_generate=False,
                    generation_config=gen_cfg,
                )
        return generated_tokens