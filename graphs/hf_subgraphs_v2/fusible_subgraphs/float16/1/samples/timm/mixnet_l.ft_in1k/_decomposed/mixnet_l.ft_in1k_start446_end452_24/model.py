import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_3 = in_1 = in_0 = None
        tmp_3 = torch.sigmoid(conv2d);  conv2d = None
        tmp_4 = in_2 * tmp_3;  in_2 = tmp_3 = None
        split = torch.functional.split(tmp_4, [792, 792], 1);  tmp_4 = None
        tmp_6 = split[0]
        tmp_7 = split[1];  split = None
        return (tmp_6, tmp_7)
        