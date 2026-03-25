import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.clip(in_0, 0.001, 10.0)
        return (tmp_0,)