import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.pixel_shuffle(in_0, 2)
        return (tmp_0,)