import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(tmp_1, tmp_0, None, (2, 2), (3, 3), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.nn.functional.max_pool2d(tmp_2, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
        tmp_2 = None
        return (tmp_3,)