import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1, in_2):
        tmp_2 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        tmp_3 = torch.cat((in_1, tmp_2), 1);  in_1 = tmp_2 = None
        conv2d = torch.conv2d(tmp_3, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_3 = w_1 = w_0 = None
        tmp_5 = conv2d * 0.1;  conv2d = None
        tmp_6 = tmp_5 + in_0;  tmp_5 = in_0 = None
        return (tmp_6,)
        