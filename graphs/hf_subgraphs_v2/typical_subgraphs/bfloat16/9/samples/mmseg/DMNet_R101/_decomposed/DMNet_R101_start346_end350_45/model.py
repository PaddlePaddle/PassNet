import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1):
        in_1 += in_0;  in_2 = in_1;  in_1 = in_0 = None
        tmp_3 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, 1)
        conv2d = torch.conv2d(tmp_4, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_4 = w_1 = w_0 = None
        return (conv2d, tmp_3)
        