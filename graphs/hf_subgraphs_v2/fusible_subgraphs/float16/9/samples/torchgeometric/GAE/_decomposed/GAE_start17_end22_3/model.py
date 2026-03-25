import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = in_0[0]
        tmp_1 = in_0[1];  in_0 = None
        tmp_2 = tmp_1.view((-1,))
        tmp_3 = tmp_2.expand_as(in_1);  tmp_2 = None
        tmp_4 = in_1.new_zeros((1000,));  in_1 = None
        return (tmp_1, tmp_3, tmp_4, tmp_0)
        