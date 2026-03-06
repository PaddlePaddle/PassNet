import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(8, 32, 13, 13)
        return (tmp_0,)