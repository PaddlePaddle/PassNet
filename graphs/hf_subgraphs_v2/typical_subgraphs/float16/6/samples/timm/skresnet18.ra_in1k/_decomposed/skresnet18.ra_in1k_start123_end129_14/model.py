import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_1 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        conv2d = torch.conv2d(tmp_1, in_0, None, (1, 1), (0, 0), (1, 1), 1);  tmp_1 = in_0 = None
        tmp_3 = conv2d.view(1, 2, 256, 1, 1);  conv2d = None
        tmp_4 = torch.softmax(tmp_3, dim = 1);  tmp_3 = None
        tmp_5 = in_1 * tmp_4;  in_1 = tmp_4 = None
        tmp_6 = torch.sum(tmp_5, dim = 1);  tmp_5 = None
        return (tmp_6,)
        