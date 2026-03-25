import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.conv2d(in_0, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_5 = torch.conv2d(in_0, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_3 = tmp_2 = None
        tmp_6 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
        tmp_4 = None
        tmp_7 = tmp_6 * tmp_5
        tmp_6 = tmp_5 = None
        return (tmp_7,)