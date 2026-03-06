import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(1, 7, 7, 4, 4, 128)
        return (tmp_0,)