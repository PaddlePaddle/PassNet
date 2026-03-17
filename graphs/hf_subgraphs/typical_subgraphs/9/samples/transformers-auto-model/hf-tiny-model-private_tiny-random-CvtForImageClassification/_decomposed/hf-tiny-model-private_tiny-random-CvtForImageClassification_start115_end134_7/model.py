import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = in_2.view(1, 48, 16)
        tmp_7 = tmp_6.permute(0, 2, 1)
        tmp_6 = None
        tmp_8 = torch.nn.functional.linear(in_1, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_9 = tmp_8.view(1, 64, 3, 16)
        tmp_8 = None
        tmp_10 = tmp_9.permute(0, 2, 1, 3)
        tmp_9 = None
        tmp_11 = torch.nn.functional.linear(in_0, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_12 = tmp_11.view(1, 16, 3, 16)
        tmp_11 = None
        tmp_13 = tmp_12.permute(0, 2, 1, 3)
        tmp_12 = None
        tmp_14 = torch.nn.functional.linear(tmp_7, tmp_5, tmp_4)
        tmp_7 = tmp_5 = tmp_4 = None
        tmp_15 = tmp_14.view(1, 16, 3, 16)
        tmp_14 = None
        tmp_16 = tmp_15.permute(0, 2, 1, 3)
        tmp_15 = None
        tmp_17 = torch.functional.einsum('bhlk,bhtk->bhlt', [tmp_10, tmp_13])
        tmp_10 = tmp_13 = None
        tmp_18 = tmp_17 * 0.14433756729740643
        tmp_17 = None
        tmp_19 = torch.nn.functional.softmax(tmp_18, dim=-1)
        tmp_18 = None
        tmp_20 = torch.nn.functional.dropout(tmp_19, 0.0, False, False)
        tmp_19 = None
        tmp_21 = torch.functional.einsum('bhlt,bhtv->bhlv', [tmp_20, tmp_16])
        tmp_20 = tmp_16 = None
        tmp_22 = tmp_21.permute(0, 2, 1, 3)
        tmp_21 = None
        tmp_23 = tmp_22.contiguous()
        tmp_22 = None
        tmp_24 = tmp_23.view(1, 64, 48)
        tmp_23 = None
        return (tmp_24,)