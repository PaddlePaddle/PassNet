import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor):
        tmp_2 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True);  in_0 = None
        tmp_3 = torch.nn.functional.adaptive_avg_pool2d(tmp_2, (1, 1));  tmp_2 = None
        tmp_4 = tmp_3.view(1, -1);  tmp_3 = None
        tmp_5 = torch.flatten(tmp_4, 1);  tmp_4 = None
        linear = torch.nn.functional.linear(tmp_5, w_1, w_0);  tmp_5 = w_1 = w_0 = None
        tmp_7 = linear.reshape(-1, 17, 4);  linear = None
        return (tmp_7,)
        