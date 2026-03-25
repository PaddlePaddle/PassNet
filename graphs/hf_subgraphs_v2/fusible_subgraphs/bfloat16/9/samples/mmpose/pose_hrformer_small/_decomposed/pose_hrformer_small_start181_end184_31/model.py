import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = torch.nn.functional.layer_norm(in_2, (64,), in_1, in_0, 1e-06);  in_2 = in_1 = in_0 = None
        tmp_3 = tmp_2.transpose(1, 2);  tmp_2 = None
        tmp_4 = tmp_3.reshape(1, 64, 32, 24);  tmp_3 = None
        return (tmp_4,)
        