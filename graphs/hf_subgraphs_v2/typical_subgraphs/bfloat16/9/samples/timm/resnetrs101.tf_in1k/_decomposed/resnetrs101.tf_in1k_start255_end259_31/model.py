import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1, in_2):
        conv2d = torch.conv2d(in_2, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  in_2 = w_1 = w_0 = None
        tmp_3 = conv2d.sigmoid();  conv2d = None
        tmp_4 = in_1 * tmp_3;  in_1 = tmp_3 = None
        tmp_4 += in_0;  tmp_5 = tmp_4;  tmp_4 = in_0 = None
        return (tmp_5,)
        