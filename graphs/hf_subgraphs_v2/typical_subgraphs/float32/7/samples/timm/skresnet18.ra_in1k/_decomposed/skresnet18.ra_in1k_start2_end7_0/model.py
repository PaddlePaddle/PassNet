import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
        tmp_0 = None
        tmp_2 = torch.functional.split(tmp_1, 32, 1)
        tmp_3 = tmp_2[0]
        tmp_4 = tmp_2[1]
        tmp_2 = None
        return (tmp_3, tmp_4, tmp_1)