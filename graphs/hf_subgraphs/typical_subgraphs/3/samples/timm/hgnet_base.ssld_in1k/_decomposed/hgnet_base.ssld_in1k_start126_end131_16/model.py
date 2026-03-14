import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
        tmp_3 = tmp_2.mean((2, 3), keepdim=True)
        tmp_4 = torch.conv2d(tmp_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_3 = tmp_1 = tmp_0 = None
        tmp_5 = torch.sigmoid(tmp_4)
        tmp_4 = None
        tmp_6 = torch.mul(tmp_2, tmp_5)
        tmp_2 = tmp_5 = None
        return (tmp_6,)