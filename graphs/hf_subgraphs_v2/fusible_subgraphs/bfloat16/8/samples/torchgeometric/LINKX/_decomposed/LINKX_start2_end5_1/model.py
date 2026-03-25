import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_1 = in_0.index_select(-2, in_2);  in_0 = in_2 = None
        tmp_2 = in_1.view((-1, 1));  in_1 = None
        tmp_3 = tmp_2.expand_as(tmp_1);  tmp_2 = None
        return (tmp_3, tmp_1)
        