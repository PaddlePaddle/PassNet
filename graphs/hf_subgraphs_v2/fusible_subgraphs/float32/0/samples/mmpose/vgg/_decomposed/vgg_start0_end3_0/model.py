import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = torch.conv2d(tmp_0, tmp_6, tmp_5, (1, 1), (1, 1), (1, 1), 1)
        tmp_0 = tmp_6 = tmp_5 = None
        tmp_8 = torch.nn.functional.batch_norm(tmp_7, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 1e-05)
        tmp_7 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        tmp_9 = torch.nn.functional.relu(tmp_8, inplace=True)
        tmp_8 = None
        return (tmp_9,)