import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_9.view(1, 384, 49)
        tmp_7 = tmp_6.permute(0, 2, 1)
        tmp_6 = None
        tmp_8 = torch.cat((in_6, in_8), dim=1)
        tmp_9 = torch.cat((in_6, in_7), dim=1)
        tmp_10 = torch.cat((in_6, tmp_7), dim=1)
        tmp_7 = None
        tmp_11 = torch.nn.functional.linear(tmp_8, tmp_3, tmp_2)
        tmp_8 = tmp_3 = tmp_2 = None
        tmp_12 = tmp_11.view(1, 197, 6, 64)
        tmp_11 = None
        tmp_13 = tmp_12.permute(0, 2, 1, 3)
        tmp_12 = None
        tmp_14 = torch.nn.functional.linear(tmp_9, tmp_1, tmp_0)
        tmp_9 = tmp_1 = tmp_0 = None
        tmp_15 = tmp_14.view(1, 50, 6, 64)
        tmp_14 = None
        tmp_16 = tmp_15.permute(0, 2, 1, 3)
        tmp_15 = None
        tmp_17 = torch.nn.functional.linear(tmp_10, tmp_5, tmp_4)
        tmp_10 = tmp_5 = tmp_4 = None
        tmp_18 = tmp_17.view(1, 50, 6, 64)
        tmp_17 = None
        tmp_19 = tmp_18.permute(0, 2, 1, 3)
        tmp_18 = None
        tmp_20 = torch.functional.einsum('bhlk,bhtk->bhlt', [tmp_13, tmp_16])
        tmp_13 = tmp_16 = None
        tmp_21 = tmp_20 * 0.05103103630798288
        tmp_20 = None
        tmp_22 = torch.nn.functional.softmax(tmp_21, dim=-1)
        tmp_21 = None
        tmp_23 = torch.nn.functional.dropout(tmp_22, 0.0, False, False)
        tmp_22 = None
        tmp_24 = torch.functional.einsum('bhlt,bhtv->bhlv', [tmp_23, tmp_19])
        tmp_23 = tmp_19 = None
        tmp_25 = tmp_24.permute(0, 2, 1, 3)
        tmp_24 = None
        tmp_26 = tmp_25.contiguous()
        tmp_25 = None
        tmp_27 = tmp_26.view(1, 197, 384)
        tmp_26 = None
        return (tmp_27,)