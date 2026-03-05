import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(16, 14, 14, 64)
        return (tmp_0,)