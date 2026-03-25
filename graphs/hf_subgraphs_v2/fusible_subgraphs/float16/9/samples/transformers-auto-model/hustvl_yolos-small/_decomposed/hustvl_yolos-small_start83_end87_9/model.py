import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        tmp_2 = in_3 + in_2;  in_3 = in_2 = None
        tmp_3 = in_4[1];  in_4 = None
        tmp_4 = tmp_2 + tmp_3;  tmp_2 = tmp_3 = None
        tmp_5 = torch.nn.functional.layer_norm(tmp_4, (384,), in_1, in_0, 1e-12);  in_1 = in_0 = None
        return (tmp_4, tmp_5)
        