import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.cumsum(2, dtype=torch.float32)
        return (tmp_0,)