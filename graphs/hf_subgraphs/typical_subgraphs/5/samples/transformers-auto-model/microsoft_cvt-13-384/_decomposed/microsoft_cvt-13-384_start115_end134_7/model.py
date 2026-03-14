import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_8.view(1, 192, 576)
        tmp_7 = tmp_6.permute(0, 2, 1)
        tmp_6 = None
        tmp_8 = torch.nn.functional.linear(in_7, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_9 = tmp_8.view(1, 2304, 3, 64)
        tmp_8 = None
        tmp_10 = tmp_9.permute(0, 2, 1, 3)
        tmp_9 = None
        tmp_11 = torch.nn.functional.linear(in_6, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_12 = tmp_11.view(1, 576, 3, 64)
        tmp_11 = None
        tmp_13 = tmp_12.permute(0, 2, 1, 3)
        tmp_12 = None
        tmp_14 = torch.nn.functional.linear(tmp_7, tmp_5, tmp_4)
        tmp_7 = tmp_5 = tmp_4 = None
        tmp_15 = tmp_14.view(1, 576, 3, 64)
        tmp_14 = None
        tmp_16 = tmp_15.permute(0, 2, 1, 3)
        tmp_15 = None
        tmp_17 = torch.functional.einsum('bhlk,bhtk->bhlt', [tmp_10, tmp_13])
        tmp_10 = tmp_13 = None
        tmp_18 = tmp_17 * 0.07216878364870322
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
        tmp_24 = tmp_23.view(1, 2304, 192)
        tmp_23 = None
        return (tmp_24,)