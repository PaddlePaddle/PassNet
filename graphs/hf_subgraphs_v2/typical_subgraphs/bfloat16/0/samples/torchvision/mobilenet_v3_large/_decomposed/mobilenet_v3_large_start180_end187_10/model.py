import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_4 = torch.nn.functional.hardswish(in_4, True);  in_4 = None
        tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1);  tmp_4 = None
        tmp_6 = torch.flatten(tmp_5, 1);  tmp_5 = None
        linear = torch.nn.functional.linear(tmp_6, in_1, in_0);  tmp_6 = in_1 = in_0 = None
        tmp_8 = torch.nn.functional.hardswish(linear, True);  linear = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.2, False, True);  tmp_8 = None
        linear_1 = torch.nn.functional.linear(tmp_9, in_3, in_2);  tmp_9 = in_3 = in_2 = None
        return (linear_1,)
        