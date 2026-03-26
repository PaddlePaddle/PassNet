import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = torch.conv2d(in_8, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_0 = None
        tmp_6 = in_7 + in_6
        tmp_7 = torch.cat([in_5, tmp_5], dim=1)
        tmp_5 = None
        tmp_8 = torch.cat((tmp_6, tmp_7), dim=1)
        tmp_6 = tmp_7 = None
        tmp_9 = torch.nn.functional.batch_norm(tmp_8, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 0.001)
        tmp_8 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        tmp_10 = torch.nn.functional.relu(tmp_9, inplace=False)
        tmp_9 = None
        tmp_11 = torch.nn.functional.adaptive_avg_pool2d(tmp_10, 1)
        tmp_10 = None
        return (tmp_11,)