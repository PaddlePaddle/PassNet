import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.max_pool2d(in_0, 2, 1, 0, 1, ceil_mode=False, return_indices=False)
        return (tmp_0,)