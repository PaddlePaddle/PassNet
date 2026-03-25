import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.repeat_interleave(in_0, repeats=7, dim=-2)
        return (tmp_0,)