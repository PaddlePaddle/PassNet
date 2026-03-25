import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = in_2 / 2.8284271247461903;  in_2 = None
        tmp_3 = in_0.view(-1);  in_0 = None
        tmp_4 = in_1[tmp_3];  in_1 = tmp_3 = None
        tmp_5 = tmp_4.view(4, 4, -1);  tmp_4 = None
        tmp_6 = tmp_5.permute(2, 0, 1);  tmp_5 = None
        tmp_7 = tmp_6.contiguous();  tmp_6 = None
        tmp_8 = tmp_7.unsqueeze(0);  tmp_7 = None
        tmp_9 = tmp_2 + tmp_8;  tmp_2 = tmp_8 = None
        return (tmp_9,)
        