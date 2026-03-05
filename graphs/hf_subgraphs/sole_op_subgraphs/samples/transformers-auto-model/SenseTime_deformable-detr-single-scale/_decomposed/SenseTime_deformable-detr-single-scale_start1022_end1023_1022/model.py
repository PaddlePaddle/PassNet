import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.grid_sample(in_1, in_0, mode='bilinear', padding_mode='zeros', align_corners=False)
        return (tmp_0,)