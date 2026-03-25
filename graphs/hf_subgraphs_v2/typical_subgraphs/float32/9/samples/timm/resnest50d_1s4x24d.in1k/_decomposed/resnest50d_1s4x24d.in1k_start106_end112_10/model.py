import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_3 = torch.conv2d(tmp_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 4)
        tmp_2 = tmp_1 = tmp_0 = None
        tmp_4 = torch.sigmoid(tmp_3)
        tmp_3 = None
        tmp_5 = tmp_4.view(1, -1, 1, 1)
        tmp_4 = None
        tmp_6 = in_0 * tmp_5
        tmp_5 = None
        tmp_7 = tmp_6.contiguous()
        tmp_6 = None
        return (tmp_7,)