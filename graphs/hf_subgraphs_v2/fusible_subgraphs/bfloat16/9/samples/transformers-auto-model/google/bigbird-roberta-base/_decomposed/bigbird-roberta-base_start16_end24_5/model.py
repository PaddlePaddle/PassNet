import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = in_0 - in_1;  in_0 = in_1 = None
        split = tmp_0.split(1, dim = -1);  tmp_0 = None
        tmp_2 = split[0]
        tmp_3 = split[1];  split = None
        tmp_4 = tmp_2.squeeze(-1);  tmp_2 = None
        tmp_5 = tmp_4.contiguous();  tmp_4 = None
        tmp_6 = tmp_3.squeeze(-1);  tmp_3 = None
        tmp_7 = tmp_6.contiguous();  tmp_6 = None
        return (tmp_5, tmp_7)
        