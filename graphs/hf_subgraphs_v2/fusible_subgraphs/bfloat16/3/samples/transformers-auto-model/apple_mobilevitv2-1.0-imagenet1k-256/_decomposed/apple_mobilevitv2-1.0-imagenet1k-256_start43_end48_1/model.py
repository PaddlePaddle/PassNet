import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_2 = in_1 = in_0 = None
        split = torch.functional.split(conv2d, split_size_or_sections = [1, 128, 128], dim = 1);  conv2d = None
        tmp_4 = split[0]
        tmp_5 = split[1]
        tmp_6 = split[2];  split = None
        return (tmp_5, tmp_4, tmp_6)
        