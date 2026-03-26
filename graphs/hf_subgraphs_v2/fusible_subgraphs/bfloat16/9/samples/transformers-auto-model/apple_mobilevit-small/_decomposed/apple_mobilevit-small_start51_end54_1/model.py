import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.reshape(1, 144, 256, 4);  in_0 = None
        tmp_1 = tmp_0.transpose(1, 3);  tmp_0 = None
        tmp_2 = tmp_1.reshape(4, 256, -1);  tmp_1 = None
        return (tmp_2,)
        