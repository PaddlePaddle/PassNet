import torch

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.arange(249, dtype=torch.int64)
        return (tmp_0,)