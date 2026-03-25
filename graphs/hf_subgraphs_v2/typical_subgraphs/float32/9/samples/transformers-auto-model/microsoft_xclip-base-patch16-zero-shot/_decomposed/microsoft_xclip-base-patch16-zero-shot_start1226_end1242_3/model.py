import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1, in_2, in_3):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = 1.702 * in_0
        tmp_5 = torch.sigmoid(tmp_4)
        tmp_4 = None
        tmp_6 = in_0 * tmp_5
        tmp_5 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
        tmp_6 = None
        tmp_8 = torch.nn.functional.linear(tmp_7, tmp_1, tmp_0)
        tmp_7 = tmp_1 = tmp_0 = None
        tmp_9 = in_3 + tmp_8
        tmp_8 = None
        tmp_10 = tmp_2 * tmp_9
        tmp_2 = tmp_9 = None
        tmp_11 = in_2 + tmp_10
        tmp_10 = None
        tmp_12 = in_1.norm(p=2, dim=-1, keepdim=True)
        tmp_13 = in_1 / tmp_12
        tmp_12 = None
        tmp_14 = tmp_11.norm(p=2, dim=-1, keepdim=True)
        tmp_15 = tmp_11 / tmp_14
        tmp_11 = tmp_14 = None
        tmp_16 = tmp_3.exp()
        tmp_3 = None
        tmp_17 = tmp_16 * tmp_15
        tmp_16 = None
        tmp_18 = torch.functional.einsum('bd,bkd->bk', tmp_13, tmp_17)
        tmp_17 = None
        tmp_19 = tmp_18.T
        return (tmp_13, tmp_15, tmp_18, tmp_19)