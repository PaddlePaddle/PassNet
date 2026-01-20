import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1):
        tmp_0 = torch.conv2d(in_1, w_1, w_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = torch.sigmoid(tmp_0)
        tmp_0 = None
        tmp_2 = in_0 * tmp_1
        tmp_1 = None
        tmp_3 = torch.functional.split(tmp_2, [792, 792], 1)
        tmp_2 = None
        tmp_4 = tmp_3[0]
        tmp_5 = tmp_3[1]
        tmp_3 = None
        return (tmp_4, tmp_5)