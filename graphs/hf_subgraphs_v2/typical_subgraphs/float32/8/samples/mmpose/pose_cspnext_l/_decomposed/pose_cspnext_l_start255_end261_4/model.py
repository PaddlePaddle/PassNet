import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.silu(in_3, inplace=True)
        tmp_3 = torch.cat((tmp_2, in_2), dim=1)
        tmp_2 = None
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, 1)
        tmp_5 = torch.conv2d(tmp_4, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_4 = tmp_1 = tmp_0 = None
        tmp_6 = torch.nn.functional.hardsigmoid(tmp_5, True)
        tmp_5 = None
        tmp_7 = tmp_3 * tmp_6
        tmp_3 = tmp_6 = None
        return (tmp_7,)