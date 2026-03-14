import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.view(64, -1, 12, 64)
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = in_4.transpose(-1, -2)
        tmp_6 = torch.matmul(in_5, tmp_5)
        tmp_5 = None
        tmp_7 = tmp_6 / 8.0
        tmp_6 = None
        tmp_8 = tmp_7 + in_3
        tmp_7 = None
        tmp_9 = torch.nn.functional.softmax(tmp_8, dim=-1)
        tmp_8 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.1, False, False)
        tmp_9 = None
        tmp_11 = torch.matmul(tmp_10, tmp_4)
        tmp_10 = tmp_4 = None
        tmp_12 = tmp_11.permute(0, 2, 1, 3)
        tmp_11 = None
        tmp_13 = tmp_12.contiguous()
        tmp_12 = None
        tmp_14 = tmp_13.view(64, 128, 768)
        tmp_13 = None
        return (tmp_14,)