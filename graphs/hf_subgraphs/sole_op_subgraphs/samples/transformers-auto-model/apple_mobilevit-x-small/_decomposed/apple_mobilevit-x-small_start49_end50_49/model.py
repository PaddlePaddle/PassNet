import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(1536, 2, 16, 2)
        return (tmp_0,)