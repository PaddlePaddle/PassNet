import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.sigmoid(in_0)
        tmp_1 = in_0 * tmp_0
        tmp_0 = None
        tmp_2 = torch.nn.functional.max_pool2d(tmp_1, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
        tmp_3 = torch.nn.functional.max_pool2d(tmp_1, 9, 1, 4, 1, ceil_mode=False, return_indices=False)
        tmp_4 = torch.nn.functional.max_pool2d(tmp_1, 13, 1, 6, 1, ceil_mode=False, return_indices=False)
        tmp_5 = torch.cat([tmp_1, tmp_2, tmp_3, tmp_4], dim=1)
        tmp_1 = tmp_2 = tmp_3 = tmp_4 = None
        return (tmp_5,)