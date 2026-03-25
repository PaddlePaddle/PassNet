import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        tmp_3 = in_0 * in_4;  in_0 = in_4 = None
        tmp_4 = tmp_3 + in_3;  tmp_3 = in_3 = None
        tmp_5 = torch.nn.functional.layer_norm(tmp_4, (1024,), in_2, in_1, 1e-12);  tmp_4 = in_2 = in_1 = None
        return (tmp_5,)
        