import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = torch.nn.functional.relu(in_3, inplace = True);  in_3 = None
        conv2d = torch.conv2d(tmp_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 4);  tmp_2 = in_1 = in_0 = None
        tmp_4 = torch.sigmoid(conv2d);  conv2d = None
        tmp_5 = tmp_4.view(1, -1, 1, 1);  tmp_4 = None
        tmp_6 = in_2 * tmp_5;  in_2 = tmp_5 = None
        tmp_7 = tmp_6.contiguous();  tmp_6 = None
        return (tmp_7,)
        