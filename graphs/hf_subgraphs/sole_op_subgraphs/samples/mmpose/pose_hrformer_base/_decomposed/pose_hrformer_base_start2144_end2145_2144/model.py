import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0[slice(None, None, None), slice(3, 11, None), slice(0, 6, None)]
        return (tmp_0,)