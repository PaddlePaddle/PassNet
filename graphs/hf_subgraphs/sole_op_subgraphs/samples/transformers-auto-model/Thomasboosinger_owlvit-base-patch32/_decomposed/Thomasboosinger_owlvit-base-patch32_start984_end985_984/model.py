import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.elu(in_0, 1.0, False)
        return (tmp_0,)