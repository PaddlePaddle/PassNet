import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.broadcast_to(in_0, (1, 256, 32))
        return (tmp_0,)