import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        conv2d = torch.conv2d(in_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_4 = in_1 = in_0 = None
        tmp_3 = conv2d.sigmoid();  conv2d = None
        tmp_4 = in_3 * tmp_3;  in_3 = tmp_3 = None
        tmp_4 += in_2;  tmp_5 = tmp_4;  tmp_4 = in_2 = None
        tmp_6 = torch.nn.functional.relu(tmp_5, inplace = True);  tmp_5 = None
        tmp_7 = torch.nn.functional.adaptive_avg_pool2d(tmp_6, 1);  tmp_6 = None
        tmp_8 = tmp_7.flatten(1, -1);  tmp_7 = None
        return (tmp_8,)
        