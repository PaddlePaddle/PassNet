import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor):
        tmp_2 = torch.nn.functional.avg_pool2d(in_0, 3, 1, 1, False, False, None)
        tmp_3 = tmp_2 - in_0;  tmp_2 = None
        tmp_4 = w_0.unsqueeze(-1);  w_0 = None
        tmp_5 = tmp_4.unsqueeze(-1);  tmp_4 = None
        tmp_6 = tmp_5 * tmp_3;  tmp_5 = tmp_3 = None
        tmp_7 = in_0 + tmp_6;  in_0 = tmp_6 = None
        tmp_8 = w_1.unsqueeze(-1);  w_1 = None
        tmp_9 = tmp_8.unsqueeze(-1);  tmp_8 = None
        return (tmp_7, tmp_9)
        