import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.conv2d(input=in_1, weight=in_0, groups=512)
        return (tmp_0,)