import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_3 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_4 = tmp_2.reshape(64, 256, -1)
        tmp_2 = None
        tmp_5 = tmp_3.reshape(64, 512, -1)
        tmp_3 = None
        tmp_6 = tmp_5.permute(0, 2, 1)
        tmp_5 = None
        tmp_7 = tmp_6.contiguous()
        tmp_6 = None
        tmp_8 = torch.matmul(in_0, tmp_4)
        tmp_4 = None
        tmp_9 = 0.0625 * tmp_8
        tmp_8 = None
        tmp_10 = torch.nn.functional.softmax(tmp_9, dim=-1)
        tmp_9 = None
        tmp_11 = torch.matmul(tmp_10, tmp_7)
        tmp_10 = tmp_7 = None
        tmp_12 = tmp_11.permute(0, 2, 1)
        tmp_11 = None
        tmp_13 = tmp_12.contiguous()
        tmp_12 = None
        tmp_14 = tmp_13.reshape(64, -1, 8, 8)
        tmp_13 = None
        return (tmp_14,)