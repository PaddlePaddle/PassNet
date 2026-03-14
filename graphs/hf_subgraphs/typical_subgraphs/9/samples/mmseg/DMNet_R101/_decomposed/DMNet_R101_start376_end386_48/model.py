import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = torch.nn.functional.relu(in_2, inplace=True)
        tmp_7 = tmp_6.view(1, 512, 64, 64)
        tmp_6 = None
        tmp_8 = in_0.view(512, 1, 5, 5)
        tmp_9 = torch.nn.functional.pad(tmp_7, (2, 2, 2, 2), 'constant', 0)
        tmp_7 = None
        tmp_10 = torch.conv2d(input=tmp_9, weight=tmp_8, groups=512)
        tmp_9 = tmp_8 = None
        tmp_11 = tmp_10.view(1, 512, 64, 64)
        tmp_10 = None
        tmp_12 = torch.nn.functional.batch_norm(tmp_11, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_11 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_13 = torch.nn.functional.relu(tmp_12, inplace=False)
        tmp_12 = None
        tmp_14 = torch.nn.functional.adaptive_avg_pool2d(in_1, 7)
        tmp_15 = torch.conv2d(tmp_14, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_14 = tmp_5 = tmp_4 = None
        return (tmp_15, tmp_13)