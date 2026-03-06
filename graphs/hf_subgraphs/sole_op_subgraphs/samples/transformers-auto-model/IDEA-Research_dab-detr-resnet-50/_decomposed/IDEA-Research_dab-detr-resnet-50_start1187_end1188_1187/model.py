import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        in_1[Ellipsis, slice(None, 128, None)] = in_0
        tmp_0 = in_1
        tmp_0 = None
        return ()