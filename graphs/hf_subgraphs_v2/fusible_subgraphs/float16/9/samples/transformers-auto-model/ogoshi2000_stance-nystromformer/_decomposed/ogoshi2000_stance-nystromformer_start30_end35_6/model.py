import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        conv2d = torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1), 12);  in_2 = in_0 = None
        in_1 += conv2d;  in_3 = in_1;  in_1 = conv2d = None
        tmp_3 = in_3.permute(0, 2, 1, 3);  in_3 = None
        tmp_4 = tmp_3.contiguous();  tmp_3 = None
        tmp_5 = tmp_4.view(1, 16, 768);  tmp_4 = None
        return (tmp_5,)
        