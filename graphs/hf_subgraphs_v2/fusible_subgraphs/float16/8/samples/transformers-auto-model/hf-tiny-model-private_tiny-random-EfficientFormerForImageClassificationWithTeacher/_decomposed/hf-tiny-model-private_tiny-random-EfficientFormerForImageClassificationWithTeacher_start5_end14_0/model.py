import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = torch.nn.functional.relu(in_2, inplace = False);  in_2 = None
        tmp_3 = torch.nn.functional.avg_pool2d(tmp_2, 3, 1, 1, False, False, None)
        tmp_4 = tmp_3 - tmp_2;  tmp_3 = None
        tmp_5 = in_0.unsqueeze(-1);  in_0 = None
        tmp_6 = tmp_5.unsqueeze(-1);  tmp_5 = None
        tmp_7 = tmp_6 * tmp_4;  tmp_6 = tmp_4 = None
        tmp_8 = tmp_2 + tmp_7;  tmp_2 = tmp_7 = None
        tmp_9 = in_1.unsqueeze(-1);  in_1 = None
        tmp_10 = tmp_9.unsqueeze(-1);  tmp_9 = None
        return (tmp_8, tmp_10)
        