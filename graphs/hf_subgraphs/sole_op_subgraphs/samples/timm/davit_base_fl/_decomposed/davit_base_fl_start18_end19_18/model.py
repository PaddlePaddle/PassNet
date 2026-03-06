import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(256, 144, 3, 4, 32)
        return (tmp_0,)