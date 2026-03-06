import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(960, 2, 8, 2)
        return (tmp_0,)