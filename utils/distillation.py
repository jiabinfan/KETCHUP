import torch
from torch import nn
import torch.nn.functional as F

def KLD(student_logits, teacher_logits, mask=None):
    student_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    kl_loss = F.kl_div(student_probs, teacher_probs, reduction='none')
    kl_loss = kl_sum(kl_loss, mask)
    return kl_loss

def RKL(student_logits, teacher_logits, mask=None):
    student_probs = F.softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    reverse_kl_loss = F.kl_div(teacher_log_probs, student_probs, reduction='none')
    reverse_kl_loss=kl_sum(reverse_kl_loss,mask)

    return reverse_kl_loss

def JS(s_id_s_logits, s_id_t_logits, t_id_s_logits, t_id_t_logits, s_ids, t_ids, pad_token_id):
      
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

def kl_sum(kl_loss, mask):
    if mask is not None:
        mask = mask.unsqueeze(-1)
        kl_loss = kl_loss * mask
        kl_loss = kl_loss.sum() / mask.shape[0]
    else:
        kl_loss = kl_loss.mean()  # Average over all elements if no mask is provided
    return kl_loss



