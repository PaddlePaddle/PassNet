import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.nn.functional.relu(tmp_2, inplace=True)
        tmp_2 = None
        tmp_4 = tmp_3.flatten(1, -1)
        tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.2, training=False)
        tmp_4 = None
        return (tmp_5,)