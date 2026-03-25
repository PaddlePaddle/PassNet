import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = in_0.reshape(1, 1600, 64);  in_0 = None
        tmp_1 = in_1.reshape(1, 40, 40, 64);  in_1 = None
        tmp_2 = tmp_1.permute(0, 3, 1, 2);  tmp_1 = None
        tmp_3 = tmp_2.contiguous();  tmp_2 = None
        tmp_4 = tmp_0.reshape(1, 40, 40, 64);  tmp_0 = None
        tmp_5 = tmp_4.permute(0, 3, 1, 2);  tmp_4 = None
        tmp_6 = tmp_5.contiguous();  tmp_5 = None
        return (tmp_6, tmp_3)
        