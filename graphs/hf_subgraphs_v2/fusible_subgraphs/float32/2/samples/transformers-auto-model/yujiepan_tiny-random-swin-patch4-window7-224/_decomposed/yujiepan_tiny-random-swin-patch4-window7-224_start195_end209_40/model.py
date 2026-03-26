import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2 / 2.0
        tmp_3 = tmp_0.view(-1)
        tmp_0 = None
        tmp_4 = tmp_1[tmp_3]
        tmp_1 = tmp_3 = None
        tmp_5 = tmp_4.view(49, 49, -1)
        tmp_4 = None
        tmp_6 = tmp_5.permute(2, 0, 1)
        tmp_5 = None
        tmp_7 = tmp_6.contiguous()
        tmp_6 = None
        tmp_8 = tmp_7.unsqueeze(0)
        tmp_7 = None
        tmp_9 = tmp_2 + tmp_8
        tmp_2 = tmp_8 = None
        tmp_10 = torch.nn.functional.softmax(tmp_9, dim=-1)
        tmp_9 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False)
        tmp_10 = None
        tmp_12 = torch.matmul(tmp_11, in_3)
        tmp_11 = None
        tmp_13 = tmp_12.permute(0, 2, 1, 3)
        tmp_12 = None
        tmp_14 = tmp_13.contiguous()
        tmp_13 = None
        tmp_15 = tmp_14.view((1, 49, 64))
        tmp_14 = None
        return (tmp_15,)