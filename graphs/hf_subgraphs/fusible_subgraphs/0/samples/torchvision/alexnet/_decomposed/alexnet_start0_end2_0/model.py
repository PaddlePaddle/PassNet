import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.conv2d(tmp_2, tmp_1, tmp_0, (4, 4), (2, 2), (1, 1), 1)
        tmp_2 = tmp_1 = tmp_0 = None
        tmp_4 = torch.nn.functional.relu(tmp_3, inplace=True)
        tmp_3 = None
        return (tmp_4,)