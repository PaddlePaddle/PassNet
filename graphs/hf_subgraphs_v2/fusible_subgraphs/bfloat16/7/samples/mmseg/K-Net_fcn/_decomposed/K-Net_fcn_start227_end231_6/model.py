import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(in_4, in_4, in_4, 512, 8, in_3, in_2, None, None, False, 0.0, in_1, in_0, training = False, key_padding_mask = None, need_weights = True, attn_mask = None, average_attn_weights = True, is_causal = False);  in_4 = in_3 = in_2 = in_1 = in_0 = None
        tmp_5 = multi_head_attention_forward[0];  multi_head_attention_forward = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False);  tmp_5 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False);  tmp_6 = None
        return (tmp_7,)
        