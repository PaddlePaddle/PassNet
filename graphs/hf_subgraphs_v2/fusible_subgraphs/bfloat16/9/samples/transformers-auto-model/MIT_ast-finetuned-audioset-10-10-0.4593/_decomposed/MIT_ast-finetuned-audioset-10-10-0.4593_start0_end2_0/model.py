import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = in_0.unsqueeze(1);  in_0 = None
        tmp_2 = tmp_1.transpose(2, 3);  tmp_1 = None
        return (tmp_2,)
        