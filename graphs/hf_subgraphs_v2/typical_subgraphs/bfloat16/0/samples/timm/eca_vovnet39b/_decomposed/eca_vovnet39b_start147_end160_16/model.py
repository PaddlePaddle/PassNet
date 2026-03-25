import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        tmp_3 = torch.nn.functional.relu(in_4, inplace = True);  in_4 = None
        tmp_4 = tmp_3.mean((2, 3))
        tmp_5 = tmp_4.view(1, 1, -1);  tmp_4 = None
        to = tmp_5.to(torch.bfloat16);  tmp_5 = None
        conv1d = torch.conv1d(to, in_2, None, (1,), (2,), (1,), 1);  to = in_2 = None
        tmp_7 = conv1d.sigmoid();  conv1d = None
        tmp_8 = tmp_7.view(1, -1, 1, 1);  tmp_7 = None
        tmp_9 = tmp_8.expand_as(tmp_3);  tmp_8 = None
        tmp_10 = tmp_3 * tmp_9;  tmp_3 = tmp_9 = None
        tmp_11 = tmp_10 + in_3;  tmp_10 = in_3 = None
        tmp_12 = torch.nn.functional.adaptive_avg_pool2d(tmp_11, 1);  tmp_11 = None
        tmp_13 = tmp_12.flatten(1, -1);  tmp_12 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False);  tmp_13 = None
        to_1 = tmp_14.to(torch.bfloat16);  tmp_14 = None
        linear = torch.nn.functional.linear(to_1, in_1, in_0);  to_1 = in_1 = in_0 = None
        return (linear,)
        