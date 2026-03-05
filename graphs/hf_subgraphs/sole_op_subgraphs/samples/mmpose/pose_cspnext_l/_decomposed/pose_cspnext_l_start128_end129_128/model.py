import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.hardsigmoid(in_0, True)
        return (tmp_0,)