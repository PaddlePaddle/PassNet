import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_1 = in_0 * 1000000.0;  in_0 = None
        tmp_2 = in_1 - tmp_1;  in_1 = tmp_1 = None
        split = tmp_2.split(1, dim = -1);  tmp_2 = None
        tmp_4 = split[0]
        tmp_5 = split[1];  split = None
        tmp_6 = tmp_4.squeeze(-1);  tmp_4 = None
        tmp_7 = tmp_6.contiguous();  tmp_6 = None
        tmp_8 = tmp_5.squeeze(-1);  tmp_5 = None
        tmp_9 = tmp_8.contiguous();  tmp_8 = None
        return (tmp_7, tmp_9)
        