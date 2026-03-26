import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_4 = torch.nn.functional.hardswish(in_4, True);  in_4 = None
        tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1);  tmp_4 = None
        conv2d = torch.conv2d(tmp_5, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  tmp_5 = in_3 = in_2 = None
        tmp_7 = torch.nn.functional.hardswish(conv2d, True);  conv2d = None
        tmp_8 = tmp_7.flatten(1, -1);  tmp_7 = None
        linear = torch.nn.functional.linear(tmp_8, in_1, in_0);  tmp_8 = in_1 = in_0 = None
        return (linear,)
        