import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_2 = in_1 = in_0 = None
        tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False);  in_3 = None
        tmp_4 = tmp_3 * conv2d;  tmp_3 = conv2d = None
        return (tmp_4,)
        