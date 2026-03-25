import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor):
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(in_0, in_0, in_0, 256, 4, w_3, w_2, None, None, False, 0.0, w_1, w_0, training = False, key_padding_mask = None, need_weights = False, attn_mask = None, average_attn_weights = True, is_causal = False);  in_0 = w_3 = w_2 = w_1 = w_0 = None
        getitem = multi_head_attention_forward[0];  multi_head_attention_forward = None
        return (getitem,)
        