import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.silu(in_2, inplace=True)
        tmp_3 = tmp_2 + in_1
        tmp_2 = None
        tmp_4 = torch.cat((tmp_3, in_0), dim=1)
        tmp_3 = None
        tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
        tmp_6 = torch.conv2d(tmp_5, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_5 = tmp_1 = tmp_0 = None
        tmp_7 = torch.nn.functional.hardsigmoid(tmp_6, True)
        tmp_6 = None
        tmp_8 = tmp_4 * tmp_7
        tmp_4 = tmp_7 = None
        return (tmp_8,)