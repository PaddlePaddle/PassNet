import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(0, None, 2)]
        return (tmp_0,)