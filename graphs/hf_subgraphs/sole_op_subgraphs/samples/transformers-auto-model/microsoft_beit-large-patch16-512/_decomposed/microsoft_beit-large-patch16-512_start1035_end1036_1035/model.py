import torch

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.zeros(size=(1025, 1025), dtype=torch.int64)
        return (tmp_0,)