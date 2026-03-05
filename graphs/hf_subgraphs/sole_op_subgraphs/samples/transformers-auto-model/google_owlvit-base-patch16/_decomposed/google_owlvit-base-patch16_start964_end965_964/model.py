import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.broadcast_to(in_0, (1, 2304, 768))
        return (tmp_0,)