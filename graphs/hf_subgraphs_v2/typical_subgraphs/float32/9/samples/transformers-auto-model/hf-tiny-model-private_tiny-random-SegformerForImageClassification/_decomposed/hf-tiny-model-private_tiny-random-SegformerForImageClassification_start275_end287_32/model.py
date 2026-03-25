import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.linear(in_1, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.view(1, -1, 8, 16)
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = in_0.transpose(-1, -2)
        tmp_6 = torch.matmul(in_2, tmp_5)
        tmp_5 = None
        tmp_7 = tmp_6 / 4.0
        tmp_6 = None
        tmp_8 = torch.nn.functional.softmax(tmp_7, dim=-1)
        tmp_7 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False)
        tmp_8 = None
        tmp_10 = torch.matmul(tmp_9, tmp_4)
        tmp_9 = tmp_4 = None
        tmp_11 = tmp_10.permute(0, 2, 1, 3)
        tmp_10 = None
        tmp_12 = tmp_11.contiguous()
        tmp_11 = None
        tmp_13 = tmp_12.view((1, 4, 128))
        tmp_12 = None
        return (tmp_13,)