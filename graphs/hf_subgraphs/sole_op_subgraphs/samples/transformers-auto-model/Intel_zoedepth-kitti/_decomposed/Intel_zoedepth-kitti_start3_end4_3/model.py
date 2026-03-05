import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(4, 32, 32, 1024)
        return (tmp_0,)