import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.view(128, 3, 40, 24, 24);  in_0 = None
        tmp_1 = torch.transpose(tmp_0, 1, 2);  tmp_0 = None
        tmp_2 = tmp_1.contiguous();  tmp_1 = None
        tmp_3 = tmp_2.view(128, 120, 24, 24);  tmp_2 = None
        return (tmp_3,)
        