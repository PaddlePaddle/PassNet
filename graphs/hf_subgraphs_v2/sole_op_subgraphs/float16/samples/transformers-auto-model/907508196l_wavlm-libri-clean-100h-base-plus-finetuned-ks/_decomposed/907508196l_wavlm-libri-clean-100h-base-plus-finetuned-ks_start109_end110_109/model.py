import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, in_0, in_1, in_2, in_3):
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(in_3, in_3, in_3, 768, 12, in_1, in_0, None, None, False, 0.0, w_2, w_1, False, None, False, in_2, use_separate_proj_weight = True, q_proj_weight = w_3, k_proj_weight = w_0, v_proj_weight = w_4);  in_3 = in_1 = in_0 = w_2 = w_1 = in_2 = w_3 = w_0 = w_4 = None
        getitem = multi_head_attention_forward[0];  multi_head_attention_forward = None
        return (getitem,)
        