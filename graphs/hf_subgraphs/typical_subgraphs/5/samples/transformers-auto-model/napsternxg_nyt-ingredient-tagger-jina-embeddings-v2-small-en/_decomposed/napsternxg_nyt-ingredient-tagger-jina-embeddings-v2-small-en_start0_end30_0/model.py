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
        tmp_10 = in_10
        tmp_11 = in_11
        tmp_12 = in_12
        tmp_13 = in_13
        tmp_14 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_15 = tmp_14.to(dtype=torch.float32)
        tmp_14 = None
        tmp_16 = 1.0 - tmp_15
        tmp_15 = None
        tmp_17 = tmp_16 * -3.4028234663852886e+38
        tmp_16 = None
        tmp_18 = torch.nn.functional.embedding(tmp_1, tmp_5, 0, None, 2.0, False, False)
        tmp_1 = tmp_5 = None
        tmp_19 = torch.nn.functional.embedding(tmp_13, tmp_4, None, None, 2.0, False, False)
        tmp_13 = tmp_4 = None
        tmp_20 = tmp_18 + tmp_19
        tmp_18 = tmp_19 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (512,), tmp_3, tmp_2, 1e-12)
        tmp_20 = tmp_3 = tmp_2 = None
        tmp_22 = torch.nn.functional.dropout(tmp_21, 0.1, False, False)
        tmp_21 = None
        tmp_23 = tmp_6[slice(None, None, None), slice(None, None, None), slice(None, 512, None), slice(None, 512, None)]
        tmp_6 = None
        tmp_24 = torch.nn.functional.linear(tmp_22, tmp_10, tmp_9)
        tmp_10 = tmp_9 = None
        tmp_25 = torch.nn.functional.linear(tmp_22, tmp_8, tmp_7)
        tmp_8 = tmp_7 = None
        tmp_26 = tmp_25.view((4, 512, 8, 64))
        tmp_25 = None
        tmp_27 = tmp_26.permute(0, 2, 1, 3)
        tmp_26 = None
        tmp_28 = torch.nn.functional.linear(tmp_22, tmp_12, tmp_11)
        tmp_12 = tmp_11 = None
        tmp_29 = tmp_28.view((4, 512, 8, 64))
        tmp_28 = None
        tmp_30 = tmp_29.permute(0, 2, 1, 3)
        tmp_29 = None
        tmp_31 = tmp_24.view((4, 512, 8, 64))
        tmp_24 = None
        tmp_32 = tmp_31.permute(0, 2, 1, 3)
        tmp_31 = None
        tmp_33 = tmp_27.transpose(-1, -2)
        tmp_27 = None
        tmp_34 = torch.matmul(tmp_32, tmp_33)
        tmp_32 = tmp_33 = None
        tmp_35 = tmp_34 / 8.0
        tmp_34 = None
        tmp_36 = tmp_35 + tmp_17
        tmp_35 = None
        tmp_37 = tmp_36 + tmp_23
        tmp_36 = None
        tmp_38 = torch.nn.functional.softmax(tmp_37, dim=-1)
        tmp_37 = None
        tmp_39 = torch.nn.functional.dropout(tmp_38, 0.0, False, False)
        tmp_38 = None
        tmp_40 = torch.matmul(tmp_39, tmp_30)
        tmp_39 = tmp_30 = None
        tmp_41 = tmp_40.permute(0, 2, 1, 3)
        tmp_40 = None
        tmp_42 = tmp_41.contiguous()
        tmp_41 = None
        tmp_43 = tmp_42.view((4, 512, 512))
        tmp_42 = None
        return (tmp_22, tmp_17, tmp_23, tmp_43)