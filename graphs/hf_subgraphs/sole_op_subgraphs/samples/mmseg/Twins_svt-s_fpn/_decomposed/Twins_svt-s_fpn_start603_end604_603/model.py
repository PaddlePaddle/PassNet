import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(1, 3, 3, 7, 7, 512)
        return (tmp_0,)