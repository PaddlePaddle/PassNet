import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_3, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.view(16, -1, 8, 64)
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = in_2.transpose(-1, -2)
        tmp_6 = torch.matmul(in_4, tmp_5)
        tmp_5 = None
        tmp_7 = tmp_6 / 8.0
        tmp_6 = None
        tmp_8 = torch.nn.functional.softmax(tmp_7, dim=-1)
        tmp_7 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
        tmp_8 = None
        tmp_10 = torch.matmul(tmp_9, tmp_4)
        tmp_9 = tmp_4 = None
        tmp_11 = tmp_10.permute(0, 2, 1, 3)
        tmp_10 = None
        tmp_12 = tmp_11.contiguous()
        tmp_11 = None
        tmp_13 = tmp_12.view((16, 256, 512))
        tmp_12 = None
        return (tmp_13,)