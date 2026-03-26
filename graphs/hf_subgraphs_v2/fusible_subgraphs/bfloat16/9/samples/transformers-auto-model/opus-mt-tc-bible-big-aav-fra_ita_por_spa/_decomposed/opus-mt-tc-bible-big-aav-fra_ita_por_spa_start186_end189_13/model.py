import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_3 = in_3 + in_4;  in_3 = in_4 = None
        tmp_4 = torch.nn.functional.layer_norm(tmp_3, (1024,), in_2, in_1, 1e-05);  tmp_3 = in_2 = in_1 = None
        tmp_5 = in_0.view(-1, 1);  in_0 = tmp_5 = None
        return (tmp_4,)
        