import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        tmp_2 = torch.nn.functional.relu(in_4, inplace = True);  in_4 = None
        tmp_3 = torch.cat((in_5, in_3, tmp_2), 1);  in_5 = in_3 = tmp_2 = None
        conv2d = torch.conv2d(tmp_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_3 = in_1 = in_0 = None
        tmp_5 = conv2d * 0.17;  conv2d = None
        tmp_6 = tmp_5 + in_2;  tmp_5 = in_2 = None
        return (tmp_6,)
        