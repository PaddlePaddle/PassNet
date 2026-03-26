import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.unfold(in_0, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
        return (tmp_0,)