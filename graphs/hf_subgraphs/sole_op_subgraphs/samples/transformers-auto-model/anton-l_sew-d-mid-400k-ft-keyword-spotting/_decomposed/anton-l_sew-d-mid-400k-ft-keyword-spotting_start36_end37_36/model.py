import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.avg_pool1d(in_0, (2,), (2,), (0,), False, True)
        return (tmp_0,)