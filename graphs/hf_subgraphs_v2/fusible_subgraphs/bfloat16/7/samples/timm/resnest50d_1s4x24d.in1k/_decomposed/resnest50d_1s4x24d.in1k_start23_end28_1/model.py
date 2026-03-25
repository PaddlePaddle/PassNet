import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4);  in_3 = in_1 = in_0 = None
        tmp_3 = torch.sigmoid(conv2d);  conv2d = None
        tmp_4 = tmp_3.view(1, -1, 1, 1);  tmp_3 = None
        tmp_5 = in_2 * tmp_4;  in_2 = tmp_4 = None
        tmp_6 = tmp_5.contiguous();  tmp_5 = None
        return (tmp_6,)
        