import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(1, 10, 10, 7, 7, 128)
        return (tmp_0,)