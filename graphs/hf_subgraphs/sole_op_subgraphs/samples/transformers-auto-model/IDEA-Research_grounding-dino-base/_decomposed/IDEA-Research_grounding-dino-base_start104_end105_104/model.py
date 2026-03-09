import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        in_0[Ellipsis, slice(None, 7, None)] = in_1
        tmp_0 = in_0
        tmp_0 = None
        return ()