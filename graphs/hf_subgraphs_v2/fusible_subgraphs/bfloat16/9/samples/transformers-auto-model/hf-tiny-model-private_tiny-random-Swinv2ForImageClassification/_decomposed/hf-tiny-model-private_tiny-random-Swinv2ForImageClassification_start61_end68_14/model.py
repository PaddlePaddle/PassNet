import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = torch.nn.functional.layer_norm(in_3, (16,), in_1, in_0, 1e-05);  in_3 = in_1 = in_0 = None
        tmp_3 = in_2 + tmp_2;  in_2 = tmp_2 = None
        tmp_4 = tmp_3.view(1, 16, 16, 16);  tmp_3 = None
        tmp_5 = tmp_4[(slice(None, None, None), slice(0, None, 2), slice(0, None, 2), slice(None, None, None))]
        tmp_6 = tmp_4[(slice(None, None, None), slice(1, None, 2), slice(0, None, 2), slice(None, None, None))]
        tmp_7 = tmp_4[(slice(None, None, None), slice(0, None, 2), slice(1, None, 2), slice(None, None, None))]
        tmp_8 = tmp_4[(slice(None, None, None), slice(1, None, 2), slice(1, None, 2), slice(None, None, None))];  tmp_4 = None
        return (tmp_5, tmp_6, tmp_7, tmp_8)
        