import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        tmp_6 = torch.nn.functional.hardtanh(in_6, 0.0, 6.0, True);  in_6 = None
        conv2d = torch.conv2d(tmp_6, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_6 = in_1 = in_0 = None
        tmp_8 = torch.flatten(conv2d, 2);  conv2d = None
        linear = torch.nn.functional.linear(tmp_8, in_3, in_2);  in_3 = in_2 = None
        linear_1 = torch.nn.functional.linear(tmp_8, in_5, in_4);  tmp_8 = in_5 = in_4 = None
        return (linear, linear_1)
        