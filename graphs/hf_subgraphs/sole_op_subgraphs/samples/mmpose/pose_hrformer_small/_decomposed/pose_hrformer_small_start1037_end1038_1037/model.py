import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(1, 10, 7, 7, 7, 32)
        return (tmp_0,)