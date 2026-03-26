import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.reshape(288, 2, 2, 2);  in_0 = None
        tmp_1 = tmp_0.transpose(1, 2);  tmp_0 = None
        tmp_2 = tmp_1.reshape(1, 144, 4, 4);  tmp_1 = None
        return (tmp_2,)
        