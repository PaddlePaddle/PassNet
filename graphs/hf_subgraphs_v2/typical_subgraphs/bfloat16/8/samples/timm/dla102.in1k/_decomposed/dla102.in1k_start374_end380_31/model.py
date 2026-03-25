import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        in_3 += in_2;  in_4 = in_3;  in_3 = in_2 = None
        tmp_3 = torch.nn.functional.relu(in_4, inplace = True);  in_4 = None
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, 1);  tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.0, False, False);  tmp_4 = None
        conv2d = torch.conv2d(tmp_5, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_5 = in_1 = in_0 = None
        tmp_7 = conv2d.flatten(1, -1);  conv2d = None
        return (tmp_7,)
        