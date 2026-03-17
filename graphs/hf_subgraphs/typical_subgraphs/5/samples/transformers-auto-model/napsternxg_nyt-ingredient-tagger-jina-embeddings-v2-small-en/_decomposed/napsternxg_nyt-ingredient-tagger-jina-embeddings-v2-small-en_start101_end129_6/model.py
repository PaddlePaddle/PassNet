import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = in_8
        tmp_9 = in_9
        tmp_10 = in_13[slice(None, None, None), slice(None, None, None), slice(None, 2048, None)]
        tmp_11 = in_13[slice(None, None, None), slice(None, None, None), slice(2048, None, None)]
        tmp_12 = torch.nn.functional.gelu(tmp_10, approximate='none')
        tmp_10 = None
        tmp_13 = tmp_12 * tmp_11
        tmp_12 = tmp_11 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.1, False, False)
        tmp_13 = None
        tmp_15 = torch.nn.functional.linear(tmp_14, tmp_3, tmp_2)
        tmp_14 = tmp_3 = tmp_2 = None
        tmp_16 = tmp_15 + in_12
        tmp_15 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (512,), tmp_1, tmp_0, 1e-12)
        tmp_16 = tmp_1 = tmp_0 = None
        tmp_18 = torch.nn.functional.linear(tmp_17, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_19 = torch.nn.functional.linear(tmp_17, tmp_5, tmp_4)
        tmp_5 = tmp_4 = None
        tmp_20 = tmp_19.view((4, 512, 8, 64))
        tmp_19 = None
        tmp_21 = tmp_20.permute(0, 2, 1, 3)
        tmp_20 = None
        tmp_22 = torch.nn.functional.linear(tmp_17, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_23 = tmp_22.view((4, 512, 8, 64))
        tmp_22 = None
        tmp_24 = tmp_23.permute(0, 2, 1, 3)
        tmp_23 = None
        tmp_25 = tmp_18.view((4, 512, 8, 64))
        tmp_18 = None
        tmp_26 = tmp_25.permute(0, 2, 1, 3)
        tmp_25 = None
        tmp_27 = tmp_21.transpose(-1, -2)
        tmp_21 = None
        tmp_28 = torch.matmul(tmp_26, tmp_27)
        tmp_26 = tmp_27 = None
        tmp_29 = tmp_28 / 8.0
        tmp_28 = None
        tmp_30 = tmp_29 + in_10
        tmp_29 = None
        tmp_31 = tmp_30 + in_11
        tmp_30 = None
        tmp_32 = torch.nn.functional.softmax(tmp_31, dim=-1)
        tmp_31 = None
        tmp_33 = torch.nn.functional.dropout(tmp_32, 0.0, False, False)
        tmp_32 = None
        tmp_34 = torch.matmul(tmp_33, tmp_24)
        tmp_33 = tmp_24 = None
        tmp_35 = tmp_34.permute(0, 2, 1, 3)
        tmp_34 = None
        tmp_36 = tmp_35.contiguous()
        tmp_35 = None
        tmp_37 = tmp_36.view((4, 512, 512))
        tmp_36 = None
        return (tmp_17, tmp_37)