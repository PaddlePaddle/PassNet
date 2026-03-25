import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.relu(in_2, inplace=True)
        tmp_3 = torch.conv2d(tmp_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_2 = tmp_1 = tmp_0 = None
        tmp_4 = tmp_3.view(24, 1, 2, -1)
        tmp_3 = None
        tmp_5 = tmp_4.transpose(1, 2)
        tmp_4 = None
        tmp_6 = torch.nn.functional.softmax(tmp_5, dim=1)
        tmp_5 = None
        tmp_7 = tmp_6.reshape(24, -1)
        tmp_6 = None
        tmp_8 = tmp_7.view(24, -1, 1, 1)
        tmp_7 = None
        tmp_9 = tmp_8.view(24, 2, -1, 1, 1)
        tmp_8 = None
        tmp_10 = tmp_9 * in_3
        tmp_9 = None
        tmp_11 = torch.sum(tmp_10, dim=1)
        tmp_10 = None
        tmp_12 = tmp_11.contiguous()
        tmp_11 = None
        tmp_13 = torch.nn.functional.avg_pool2d(tmp_12, 3, 2, 1, False, True, None)
        tmp_12 = None
        return (tmp_13,)