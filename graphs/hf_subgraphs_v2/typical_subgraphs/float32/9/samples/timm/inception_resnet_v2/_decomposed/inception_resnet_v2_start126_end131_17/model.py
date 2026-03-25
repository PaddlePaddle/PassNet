import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0, in_1, in_2, in_3):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.relu(in_2, inplace=True)
        tmp_3 = torch.cat((in_3, in_1, tmp_2), 1)
        tmp_2 = None
        tmp_4 = torch.conv2d(tmp_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_3 = tmp_1 = tmp_0 = None
        tmp_5 = tmp_4 * 0.17
        tmp_4 = None
        tmp_6 = tmp_5 + in_0
        tmp_5 = None
        return (tmp_6,)