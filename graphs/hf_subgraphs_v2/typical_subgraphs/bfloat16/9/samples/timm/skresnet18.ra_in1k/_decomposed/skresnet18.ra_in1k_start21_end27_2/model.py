import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0, in_1):
        tmp_1 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        conv2d = torch.conv2d(tmp_1, w_0, None, (1, 1), (0, 0), (1, 1), 1);  tmp_1 = w_0 = None
        tmp_3 = conv2d.view(1, 2, 64, 1, 1);  conv2d = None
        tmp_4 = torch.softmax(tmp_3, dim = 1);  tmp_3 = None
        tmp_5 = in_0 * tmp_4;  in_0 = tmp_4 = None
        tmp_6 = torch.sum(tmp_5, dim = 1);  tmp_5 = None
        return (tmp_6,)
        