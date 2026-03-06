import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(1, 19, 19, 7, 7, 64)
        return (tmp_0,)