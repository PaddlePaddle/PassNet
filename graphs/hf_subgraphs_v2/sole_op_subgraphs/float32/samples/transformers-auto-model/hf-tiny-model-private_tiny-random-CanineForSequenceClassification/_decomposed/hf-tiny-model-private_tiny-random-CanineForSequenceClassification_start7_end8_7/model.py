import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.max_pool1d(in_0, 4, 4, 0, 1, ceil_mode=False, return_indices=False)
        return (tmp_0,)