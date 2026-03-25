import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.conv2d(in_0, in_1, padding=0)
        return (tmp_0,)