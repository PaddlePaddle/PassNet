import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        tmp_3 = torch.nn.functional.relu(in_4, inplace = False);  in_4 = None
        tmp_4 = tmp_3 + in_3;  tmp_3 = in_3 = None
        tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1);  tmp_4 = None
        conv2d = torch.conv2d(tmp_5, in_2, None, (1, 1), (0, 0), (1, 1), 1);  tmp_5 = in_2 = None
        tmp_7 = torch.nn.functional.relu(conv2d, inplace = False);  conv2d = None
        tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False);  tmp_7 = None
        tmp_9 = tmp_8.flatten(1, -1);  tmp_8 = None
        linear = torch.nn.functional.linear(tmp_9, in_1, in_0);  tmp_9 = in_1 = in_0 = None
        return (linear,)
        