import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        in_1[Ellipsis, slice(128, None, None)] = in_0
        tmp_0 = in_1
        tmp_0 = None
        return ()