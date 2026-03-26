import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0[slice(None, None, None), slice(3, 67, None), slice(0, 48, None)]
        return (tmp_0,)