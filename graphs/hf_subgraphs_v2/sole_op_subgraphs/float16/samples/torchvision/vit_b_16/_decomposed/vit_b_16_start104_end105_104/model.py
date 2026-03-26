import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor):
        _native_multi_head_attention = torch._native_multi_head_attention(in_0, in_0, in_0, 768, 12, w_3, w_2, w_1, w_0, None, False, True, None);  in_0 = w_3 = w_2 = w_1 = w_0 = None
        getitem = _native_multi_head_attention[0];  _native_multi_head_attention = None
        return (getitem,)
        