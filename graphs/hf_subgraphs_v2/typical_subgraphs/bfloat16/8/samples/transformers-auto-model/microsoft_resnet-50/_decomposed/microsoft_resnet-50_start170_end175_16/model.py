import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        in_3 += in_2;  in_4 = in_3;  in_3 = in_2 = None
        tmp_3 = torch.nn.functional.relu(in_4, inplace = False);  in_4 = None
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, (1, 1));  tmp_3 = None
        tmp_5 = tmp_4.flatten(1, -1);  tmp_4 = None
        linear = torch.nn.functional.linear(tmp_5, in_1, in_0);  tmp_5 = in_1 = in_0 = None
        return (linear,)
        