import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0, in_1, in_2, in_3):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.linear(in_1, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.view((1, -1, 20, 64))
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = in_3 * 0.125
        tmp_6 = in_2.transpose(-1, -2)
        tmp_7 = torch.matmul(tmp_5, tmp_6)
        tmp_5 = tmp_6 = None
        tmp_8 = tmp_7 + in_0
        tmp_7 = None
        tmp_9 = torch.nn.functional.softmax(tmp_8, dim=-1)
        tmp_8 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.0, False, False)
        tmp_9 = None
        tmp_11 = tmp_10.to(torch.float32)
        tmp_10 = None
        tmp_12 = torch.matmul(tmp_11, tmp_4)
        tmp_11 = tmp_4 = None
        tmp_13 = tmp_12.permute(0, 2, 1, 3)
        tmp_12 = None
        tmp_14 = tmp_13.contiguous()
        tmp_13 = None
        tmp_15 = tmp_14.view((1, 11, 1280))
        tmp_14 = None
        return (tmp_15,)