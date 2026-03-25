import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.max_pool2d(in_0, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
        tmp_1 = torch.functional.split(tmp_0, 32, 1)
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_1 = None
        return (tmp_2, tmp_3, tmp_0)