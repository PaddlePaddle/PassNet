import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0[slice(None, 3999, None), slice(None, None, None)]
        return (tmp_0,)