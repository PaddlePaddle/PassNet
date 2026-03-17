import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.hardtanh(in_2, 0.0, 6.0, True)
        tmp_3 = torch.nn.functional.adaptive_avg_pool2d(tmp_2, (1, 1))
        tmp_2 = None
        tmp_4 = tmp_3.view(2, -1)
        tmp_3 = None
        tmp_5 = torch.flatten(tmp_4, 1)
        tmp_4 = None
        tmp_6 = torch.nn.functional.linear(tmp_5, tmp_1, tmp_0)
        tmp_5 = tmp_1 = tmp_0 = None
        tmp_7 = tmp_6.reshape(-1, 17, 4)
        tmp_6 = None
        return (tmp_7,)