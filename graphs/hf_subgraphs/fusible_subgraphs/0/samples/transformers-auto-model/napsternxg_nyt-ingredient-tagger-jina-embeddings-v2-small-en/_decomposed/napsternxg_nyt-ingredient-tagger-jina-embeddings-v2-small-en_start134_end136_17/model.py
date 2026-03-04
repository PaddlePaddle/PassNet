import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0[slice(None, None, None), slice(None, None, None), slice(None, 2048, None)]
        tmp_1 = in_0[slice(None, None, None), slice(None, None, None), slice(2048, None, None)]
        return (tmp_0, tmp_1)