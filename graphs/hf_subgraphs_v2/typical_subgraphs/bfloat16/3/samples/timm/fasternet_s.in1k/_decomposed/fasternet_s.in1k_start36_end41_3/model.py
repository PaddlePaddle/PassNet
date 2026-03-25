import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        split = torch.functional.split(in_1, [128, 384], dim = 1);  in_1 = None
        tmp_2 = split[0]
        tmp_3 = split[1];  split = None
        conv2d = torch.conv2d(tmp_2, in_0, None, (1, 1), (1, 1), (1, 1), 1);  tmp_2 = in_0 = None
        tmp_5 = torch.cat((conv2d, tmp_3), 1);  conv2d = tmp_3 = None
        return (tmp_5,)
        