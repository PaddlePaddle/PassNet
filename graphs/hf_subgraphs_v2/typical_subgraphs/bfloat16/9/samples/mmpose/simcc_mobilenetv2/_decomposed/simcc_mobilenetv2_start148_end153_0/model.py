import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, in_0 : torch.Tensor):
        tmp_6 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True);  in_0 = None
        conv2d = torch.conv2d(tmp_6, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_6 = w_1 = w_0 = None
        tmp_8 = torch.flatten(conv2d, 2);  conv2d = None
        linear = torch.nn.functional.linear(tmp_8, w_3, w_2);  w_3 = w_2 = None
        linear_1 = torch.nn.functional.linear(tmp_8, w_5, w_4);  tmp_8 = w_5 = w_4 = None
        return (linear, linear_1)
        