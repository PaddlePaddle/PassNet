import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.nn.functional.hardswish(in_0, True)
        tmp_5 = tmp_4.mean((2, 3), keepdim=True)
        tmp_6 = torch.conv2d(tmp_5, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_5 = tmp_3 = tmp_2 = None
        tmp_7 = torch.nn.functional.relu(tmp_6, inplace=True)
        tmp_6 = None
        tmp_8 = torch.conv2d(tmp_7, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_7 = tmp_1 = tmp_0 = None
        tmp_9 = torch.nn.functional.hardsigmoid(tmp_8, False)
        tmp_8 = None
        tmp_10 = tmp_4 * tmp_9
        tmp_4 = tmp_9 = None
        return (tmp_10,)