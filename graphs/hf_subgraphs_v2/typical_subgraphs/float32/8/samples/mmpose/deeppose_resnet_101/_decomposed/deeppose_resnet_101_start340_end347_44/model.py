import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        in_3 += in_2
        tmp_2 = in_3
        tmp_3 = torch.nn.functional.relu(tmp_2, inplace=True)
        tmp_2 = None
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, (1, 1))
        tmp_3 = None
        tmp_5 = tmp_4.view(128, -1)
        tmp_4 = None
        tmp_6 = torch.flatten(tmp_5, 1)
        tmp_5 = None
        tmp_7 = torch.nn.functional.linear(tmp_6, tmp_1, tmp_0)
        tmp_6 = tmp_1 = tmp_0 = None
        tmp_8 = tmp_7.reshape(-1, 16, 2)
        tmp_7 = None
        return (tmp_8,)