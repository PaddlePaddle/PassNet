import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.relu(in_4, inplace=True)
        tmp_3 = torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'nearest', None)
        tmp_2 = None
        tmp_4 = in_3 + tmp_3
        tmp_3 = None
        tmp_5 = torch.conv2d(tmp_4, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_4 = tmp_1 = tmp_0 = None
        tmp_6 = tmp_5.permute(0, 2, 3, 1)
        tmp_5 = None
        tmp_7 = tmp_6.reshape(24, -1, 1)
        tmp_6 = None
        tmp_8 = torch.nn.functional.sigmoid(tmp_7)
        tmp_7 = None
        tmp_9 = torch.matmul(tmp_8, in_2)
        tmp_8 = None
        tmp_10 = tmp_9.permute(0, 2, 1)
        tmp_9 = None
        tmp_11 = tmp_10.contiguous()
        tmp_10 = None
        tmp_12 = tmp_11.view(24, 512, 64, 128)
        tmp_11 = None
        return (tmp_12,)