import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_3 = torch.conv2d(tmp_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_2 = tmp_1 = tmp_0 = None
        tmp_4 = tmp_3.sigmoid()
        tmp_3 = None
        tmp_5 = in_1 * tmp_4
        tmp_4 = None
        tmp_6 = torch.nn.functional.hardtanh(tmp_5, 0.0, 6.0, False)
        tmp_5 = None
        return (tmp_6,)