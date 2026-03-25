import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_3 = in_1 = in_0 = None
        tmp_3 = conv2d.sigmoid();  conv2d = None
        tmp_4 = in_2 * tmp_3;  in_2 = tmp_3 = None
        tmp_5 = torch.nn.functional.relu(tmp_4, inplace = True);  tmp_4 = None
        return (tmp_5,)
        