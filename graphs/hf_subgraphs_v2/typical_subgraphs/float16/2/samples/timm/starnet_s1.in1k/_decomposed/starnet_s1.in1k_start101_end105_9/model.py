import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        conv2d = torch.conv2d(in_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_1 = in_0 = None
        conv2d_1 = torch.conv2d(in_4, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  in_4 = in_3 = in_2 = None
        tmp_6 = torch.nn.functional.hardtanh(conv2d, 0.0, 6.0, False);  conv2d = None
        tmp_7 = tmp_6 * conv2d_1;  tmp_6 = conv2d_1 = None
        return (tmp_7,)
        