import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(35, 24, 24, 1024)
        return (tmp_0,)