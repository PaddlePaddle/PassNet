import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        in_0[slice(None, None, None), slice(None, None, None), slice(None, None, None), 0] = in_1
        tmp_0 = in_0
        tmp_0 = None
        return ()