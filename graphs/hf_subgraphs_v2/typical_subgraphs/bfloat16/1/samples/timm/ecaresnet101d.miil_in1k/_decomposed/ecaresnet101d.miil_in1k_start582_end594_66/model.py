import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        tmp_3 = in_4.mean((2, 3))
        tmp_4 = tmp_3.view(1, 1, -1);  tmp_3 = None
        conv1d = torch.conv1d(tmp_4, in_2, None, (1,), (3,), (1,), 1);  tmp_4 = in_2 = None
        tmp_6 = conv1d.sigmoid();  conv1d = None
        tmp_7 = tmp_6.view(1, -1, 1, 1);  tmp_6 = None
        tmp_8 = tmp_7.expand_as(in_4);  tmp_7 = None
        tmp_9 = in_4 * tmp_8;  in_4 = tmp_8 = None
        tmp_9 += in_3;  tmp_10 = tmp_9;  tmp_9 = in_3 = None
        tmp_11 = torch.nn.functional.relu(tmp_10, inplace = True);  tmp_10 = None
        tmp_12 = torch.nn.functional.adaptive_avg_pool2d(tmp_11, 1);  tmp_11 = None
        tmp_13 = tmp_12.flatten(1, -1);  tmp_12 = None
        linear = torch.nn.functional.linear(tmp_13, in_1, in_0);  tmp_13 = in_1 = in_0 = None
        return (linear,)
        