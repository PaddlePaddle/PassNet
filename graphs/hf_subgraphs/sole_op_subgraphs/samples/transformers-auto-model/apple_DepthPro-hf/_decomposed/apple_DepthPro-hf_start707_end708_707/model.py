import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0[slice(None, None, None), slice(None, None, None), slice(0, 21, None), slice(3, 21, None)]
        return (tmp_0,)