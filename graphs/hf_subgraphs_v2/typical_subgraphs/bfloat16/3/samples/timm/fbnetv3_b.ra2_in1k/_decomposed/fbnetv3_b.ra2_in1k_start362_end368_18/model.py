import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_3 = torch.nn.functional.hardswish(in_3, True);  in_3 = None
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, 1);  tmp_3 = None
        conv2d = torch.conv2d(tmp_4, in_2, None, (1, 1), (0, 0), (1, 1), 1);  tmp_4 = in_2 = None
        tmp_6 = torch.nn.functional.hardswish(conv2d, True);  conv2d = None
        tmp_7 = tmp_6.flatten(1, -1);  tmp_6 = None
        linear = torch.nn.functional.linear(tmp_7, in_1, in_0);  tmp_7 = in_1 = in_0 = None
        return (linear,)
        