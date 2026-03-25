import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_3 = tmp_2.mean((2, 3), keepdim=True)
        tmp_4 = torch.conv2d(tmp_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_3 = tmp_1 = tmp_0 = None
        tmp_5 = torch.nn.functional.hardsigmoid(tmp_4, False)
        tmp_4 = None
        tmp_6 = tmp_2 * tmp_5
        tmp_2 = tmp_5 = None
        tmp_7 = torch.nn.functional.max_pool2d(tmp_6, 3, 2, 0, 1, ceil_mode=True, return_indices=False)
        tmp_6 = None
        return (tmp_7,)