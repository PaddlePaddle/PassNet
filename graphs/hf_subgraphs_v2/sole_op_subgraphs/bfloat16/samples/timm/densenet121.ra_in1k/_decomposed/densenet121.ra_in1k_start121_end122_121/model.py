import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
        tmp_0 = torch.cat([in_0, in_7, in_8, in_9, in_10, in_1, in_2, in_3, in_4, in_5, in_6], 1);  in_0 = in_7 = in_8 = in_9 = in_10 = in_1 = in_2 = in_3 = in_4 = in_5 = in_6 = None
        return (tmp_0,)
        