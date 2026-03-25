import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor):
        tmp_4 = torch.nn.functional.hardswish(in_0, True);  in_0 = None
        tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1);  tmp_4 = None
        tmp_6 = torch.flatten(tmp_5, 1);  tmp_5 = None
        linear = torch.nn.functional.linear(tmp_6, w_1, w_0);  tmp_6 = w_1 = w_0 = None
        tmp_8 = torch.nn.functional.hardswish(linear, True);  linear = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.2, False, True);  tmp_8 = None
        linear_1 = torch.nn.functional.linear(tmp_9, w_3, w_2);  tmp_9 = w_3 = w_2 = None
        return (linear_1,)
        