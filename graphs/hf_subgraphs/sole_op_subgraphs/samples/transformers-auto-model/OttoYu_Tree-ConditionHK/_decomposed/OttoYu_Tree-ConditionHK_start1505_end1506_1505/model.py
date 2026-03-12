import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.adaptive_avg_pool1d(in_0, 1)
        return (tmp_0,)