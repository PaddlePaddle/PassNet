import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.zeros_like(in_0, dtype=torch.float32)
        return (tmp_0,)