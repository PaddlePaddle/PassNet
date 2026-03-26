import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor):
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(in_5, in_4, in_4, 768, 12, in_3, in_2, None, None, False, 0.0, in_1, in_0, training = False, key_padding_mask = None, need_weights = True, attn_mask = None, average_attn_weights = True, is_causal = False);  in_5 = in_4 = in_3 = in_2 = in_1 = in_0 = None
        tmp_5 = multi_head_attention_forward[0];  multi_head_attention_forward = None
        tmp_6 = tmp_5.transpose(1, 0);  tmp_5 = None
        return (tmp_6,)
        