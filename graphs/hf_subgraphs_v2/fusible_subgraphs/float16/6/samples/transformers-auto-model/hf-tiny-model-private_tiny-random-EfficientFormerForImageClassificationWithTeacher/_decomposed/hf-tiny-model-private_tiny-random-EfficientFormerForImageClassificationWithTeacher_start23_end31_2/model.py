import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = torch.nn.functional.avg_pool2d(in_2, 3, 1, 1, False, False, None)
        tmp_3 = tmp_2 - in_2;  tmp_2 = None
        tmp_4 = in_0.unsqueeze(-1);  in_0 = None
        tmp_5 = tmp_4.unsqueeze(-1);  tmp_4 = None
        tmp_6 = tmp_5 * tmp_3;  tmp_5 = tmp_3 = None
        tmp_7 = in_2 + tmp_6;  in_2 = tmp_6 = None
        tmp_8 = in_1.unsqueeze(-1);  in_1 = None
        tmp_9 = tmp_8.unsqueeze(-1);  tmp_8 = None
        return (tmp_7, tmp_9)
        