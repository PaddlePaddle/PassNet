import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        conv2d = torch.conv2d(in_0, in_1, padding = 0);  in_0 = in_1 = None
        tmp_1 = torch.cat([conv2d], dim = 0);  conv2d = None
        tmp_2 = tmp_1.reshape(1, 150, 128, 128);  tmp_1 = None
        tmp_3 = in_2.permute(0, 1, 3, 2);  in_2 = None
        tmp_4 = tmp_3.reshape(1, 150, 512, 1, 1);  tmp_3 = None
        return (tmp_2, tmp_4)
        