import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.sigmoid()
        tmp_2 = None
        tmp_4 = in_2 * tmp_3
        tmp_3 = None
        tmp_5 = torch.nn.functional.relu(tmp_4, inplace=True)
        tmp_4 = None
        return (tmp_5,)