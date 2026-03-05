import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.max_pool2d(in_0, 3, 2, 1, 1, ceil_mode=True, return_indices=False)
        return (tmp_0,)