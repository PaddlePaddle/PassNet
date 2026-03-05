import torch

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.arange(start=0, end=128, dtype=torch.float32)
        return (tmp_0,)