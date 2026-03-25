import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.adaptive_max_pool2d(in_0, (1, 8))
        return (tmp_0,)