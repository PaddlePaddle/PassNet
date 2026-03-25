import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = torch.nn.functional.hardtanh(in_2, 0.0, 6.0, True);  in_2 = None
        tmp_3 = torch.nn.functional.adaptive_avg_pool2d(tmp_2, (1, 1));  tmp_2 = None
        tmp_4 = tmp_3.view(16, -1);  tmp_3 = None
        tmp_5 = torch.flatten(tmp_4, 1);  tmp_4 = None
        linear = torch.nn.functional.linear(tmp_5, in_1, in_0);  tmp_5 = in_1 = in_0 = None
        tmp_7 = linear.reshape(-1, 17, 4);  linear = None
        return (tmp_7,)
        