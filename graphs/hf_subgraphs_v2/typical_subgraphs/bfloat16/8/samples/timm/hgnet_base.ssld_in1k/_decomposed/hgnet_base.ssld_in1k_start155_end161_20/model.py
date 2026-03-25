import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = torch.nn.functional.relu(in_3, inplace = False);  in_3 = None
        tmp_3 = tmp_2.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_3 = in_1 = in_0 = None
        tmp_5 = torch.sigmoid(conv2d);  conv2d = None
        tmp_6 = torch.mul(tmp_2, tmp_5);  tmp_2 = tmp_5 = None
        tmp_7 = tmp_6 + in_2;  tmp_6 = in_2 = None
        return (tmp_7,)
        