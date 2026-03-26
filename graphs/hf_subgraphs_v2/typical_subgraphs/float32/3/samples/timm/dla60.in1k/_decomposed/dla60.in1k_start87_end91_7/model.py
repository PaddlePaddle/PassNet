import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_2 = None
        tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        return (tmp_1, tmp_3, tmp_0)