import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_0 = in_1.view(-1, 1);  in_1 = None
        tmp_1 = tmp_0 * in_2;  tmp_0 = in_2 = None
        tmp_2 = in_0.view((-1, 1));  in_0 = None
        tmp_3 = tmp_2.expand_as(tmp_1);  tmp_2 = None
        tmp_4 = tmp_1.new_zeros((1000, 16))
        return (tmp_3, tmp_4, tmp_1)
        