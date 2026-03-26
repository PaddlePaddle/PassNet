import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = in_3 + in_2;  in_3 = in_2 = None
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), in_1, in_0, 1e-06);  tmp_2 = in_1 = in_0 = None
        tmp_4 = tmp_3[(slice(None, None, None), slice(0, None, None))];  tmp_3 = None
        tmp_5 = tmp_4.reshape(1, 16, 12, -1);  tmp_4 = None
        tmp_6 = tmp_5.permute(0, 3, 1, 2);  tmp_5 = None
        return (tmp_6,)
        