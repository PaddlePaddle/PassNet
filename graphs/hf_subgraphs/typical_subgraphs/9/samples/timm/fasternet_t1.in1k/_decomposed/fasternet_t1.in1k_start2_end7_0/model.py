import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0):
        tmp_0 = w_0
        tmp_1 = torch.functional.split(in_0, [16, 48], dim=1)
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_1 = None
        tmp_4 = torch.conv2d(tmp_2, tmp_0, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_2 = tmp_0 = None
        tmp_5 = torch.cat((tmp_4, tmp_3), 1)
        tmp_4 = tmp_3 = None
        return (tmp_5,)