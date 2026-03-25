import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_3 = in_1 = in_0 = None
        tmp_3 = in_2.reshape(64, 64, -1);  in_2 = None
        tmp_4 = conv2d.reshape(64, 512, -1);  conv2d = None
        tmp_5 = tmp_4.permute(0, 2, 1);  tmp_4 = None
        tmp_6 = tmp_5.contiguous();  tmp_5 = None
        return (tmp_3, tmp_6)
        