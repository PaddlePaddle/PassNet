import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1);  in_1 = in_0 = None
        tmp_2 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        tmp_3 = torch.nn.functional.adaptive_avg_pool2d(tmp_2, 1);  tmp_2 = None
        tmp_4 = tmp_3.flatten(1, -1);  tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.0, False, False);  tmp_4 = None
        return (tmp_5,)
        