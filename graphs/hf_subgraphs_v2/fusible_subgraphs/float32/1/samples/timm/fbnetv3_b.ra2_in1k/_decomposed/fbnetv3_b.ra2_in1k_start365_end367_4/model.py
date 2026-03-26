import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.hardswish(in_0, True)
        tmp_1 = tmp_0.flatten(1, -1)
        tmp_0 = None
        return (tmp_1,)