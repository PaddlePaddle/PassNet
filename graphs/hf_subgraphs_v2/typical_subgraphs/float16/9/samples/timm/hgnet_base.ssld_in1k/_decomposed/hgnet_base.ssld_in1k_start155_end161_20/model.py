import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1):
        tmp_2 = torch.nn.functional.relu(in_1, inplace = False);  in_1 = None
        tmp_3 = tmp_2.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_3, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_3 = w_1 = w_0 = None
        tmp_5 = torch.sigmoid(conv2d);  conv2d = None
        tmp_6 = torch.mul(tmp_2, tmp_5);  tmp_2 = tmp_5 = None
        tmp_7 = tmp_6 + in_0;  tmp_6 = in_0 = None
        return (tmp_7,)
        