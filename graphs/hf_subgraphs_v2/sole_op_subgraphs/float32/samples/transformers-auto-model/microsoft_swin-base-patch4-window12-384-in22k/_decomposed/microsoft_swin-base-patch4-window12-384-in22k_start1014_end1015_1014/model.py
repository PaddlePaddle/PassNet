import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.roll(in_0, shifts=(6, 6), dims=(1, 2))
        return (tmp_0,)