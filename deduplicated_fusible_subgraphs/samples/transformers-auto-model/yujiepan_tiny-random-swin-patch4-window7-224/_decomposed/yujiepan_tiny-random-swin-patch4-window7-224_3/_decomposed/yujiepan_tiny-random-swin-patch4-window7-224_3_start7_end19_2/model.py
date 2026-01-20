import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, w_0):
        tmp_0 = w_0[in_2]
        tmp_1 = tmp_0.view(49, 49, -1)
        tmp_0 = None
        tmp_2 = tmp_1.permute(2, 0, 1)
        tmp_1 = None
        tmp_3 = tmp_2.contiguous()
        tmp_2 = None
        tmp_4 = tmp_3.unsqueeze(0)
        tmp_3 = None
        tmp_5 = in_1 + tmp_4
        tmp_4 = None
        tmp_6 = torch.nn.functional.softmax(tmp_5, dim=-1)
        tmp_5 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
        tmp_6 = None
        tmp_8 = torch.matmul(tmp_7, in_0)
        tmp_7 = None
        tmp_9 = tmp_8.permute(0, 2, 1, 3)
        tmp_8 = None
        tmp_10 = tmp_9.contiguous()
        tmp_9 = None
        tmp_11 = tmp_10.view((1, 49, 64))
        tmp_10 = None
        return (tmp_11,)