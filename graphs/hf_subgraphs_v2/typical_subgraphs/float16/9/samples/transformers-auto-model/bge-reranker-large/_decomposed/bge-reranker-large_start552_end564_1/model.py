import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, in_1 : torch.Tensor):
        tmp_3 = in_1[(slice(None, None, None), 0)]
        linear = torch.nn.functional.linear(tmp_3, w_1, w_0);  tmp_3 = w_1 = w_0 = None
        tmp_5 = torch.tanh(linear);  linear = tmp_5 = None
        tmp_6 = in_0.unsqueeze(-1);  in_0 = None
        tmp_7 = tmp_6.expand((1, 10, 1024));  tmp_6 = None
        tmp_8 = tmp_7.to(torch.float32);  tmp_7 = None
        tmp_9 = in_1 * tmp_8;  in_1 = None
        tmp_10 = torch.sum(tmp_9, 1);  tmp_9 = None
        tmp_11 = tmp_8.sum(1);  tmp_8 = None
        tmp_12 = torch.clamp(tmp_11, min = 1e-09);  tmp_11 = None
        tmp_13 = tmp_10 / tmp_12;  tmp_10 = tmp_12 = None
        tmp_14 = torch.cat([tmp_13], 1);  tmp_13 = None
        return (tmp_14,)
        