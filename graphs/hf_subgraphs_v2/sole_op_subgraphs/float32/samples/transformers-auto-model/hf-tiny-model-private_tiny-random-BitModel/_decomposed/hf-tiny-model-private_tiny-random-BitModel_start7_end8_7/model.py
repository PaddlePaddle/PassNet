import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.max_pool2d(in_0, (3, 3), (2, 2), (0, 0), (1, 1), False)
        return (tmp_0,)