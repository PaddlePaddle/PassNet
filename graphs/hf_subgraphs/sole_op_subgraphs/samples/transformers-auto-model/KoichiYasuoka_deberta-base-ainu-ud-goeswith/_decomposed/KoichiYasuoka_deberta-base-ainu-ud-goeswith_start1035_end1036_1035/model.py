import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0[slice(0, 1024, None), slice(None, None, None)]
        return (tmp_0,)