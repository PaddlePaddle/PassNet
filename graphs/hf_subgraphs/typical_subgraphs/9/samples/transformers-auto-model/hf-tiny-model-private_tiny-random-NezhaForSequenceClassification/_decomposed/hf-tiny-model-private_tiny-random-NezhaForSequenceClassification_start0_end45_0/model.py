import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = w_0
        tmp_3 = w_1
        tmp_4 = w_2
        tmp_5 = w_3
        tmp_6 = w_4
        tmp_7 = w_5
        tmp_8 = w_6
        tmp_9 = w_7
        tmp_10 = w_8
        tmp_11 = w_9
        tmp_12 = w_10
        tmp_13 = in_2
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
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (32,), tmp_3, tmp_2, 1e-12)
        tmp_20 = tmp_3 = tmp_2 = None
        tmp_22 = torch.nn.functional.dropout(tmp_21, 0.1, False, False)
        tmp_21 = None
        tmp_23 = torch.nn.functional.linear(tmp_22, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_24 = torch.nn.functional.linear(tmp_22, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_25 = tmp_24.view((1, 45, 4, 8))
        tmp_24 = None
        tmp_26 = tmp_25.permute(0, 2, 1, 3)
        tmp_25 = None
        tmp_27 = torch.nn.functional.linear(tmp_22, tmp_12, tmp_11)
        tmp_12 = tmp_11 = None
        tmp_28 = tmp_27.view((1, 45, 4, 8))
        tmp_27 = None
        tmp_29 = tmp_28.permute(0, 2, 1, 3)
        tmp_28 = None
        tmp_30 = tmp_23.view((1, 45, 4, 8))
        tmp_23 = None
        tmp_31 = tmp_30.permute(0, 2, 1, 3)
        tmp_30 = None
        tmp_32 = tmp_26.transpose(-1, -2)
        tmp_26 = None
        tmp_33 = torch.matmul(tmp_31, tmp_32)
        tmp_32 = None
        tmp_34 = tmp_10[slice(None, 45, None), slice(None, 45, None), slice(None, None, None)]
        tmp_35 = tmp_31.permute(2, 0, 1, 3)
        tmp_31 = None
        tmp_36 = tmp_35.contiguous()
        tmp_35 = None
        tmp_37 = tmp_36.view(45, 4, 8)
        tmp_36 = None
        tmp_38 = tmp_34.permute(0, 2, 1)
        tmp_34 = None
        tmp_39 = torch.matmul(tmp_37, tmp_38)
        tmp_37 = tmp_38 = None
        tmp_40 = tmp_39.view(45, 1, 4, 45)
        tmp_39 = None
        tmp_41 = tmp_40.permute(1, 2, 0, 3)
        tmp_40 = None
        tmp_42 = tmp_33 + tmp_41
        tmp_33 = tmp_41 = None
        tmp_43 = tmp_42 / 2.8284271247461903
        tmp_42 = None
        tmp_44 = tmp_43 + tmp_17
        tmp_43 = None
        tmp_45 = torch.nn.functional.softmax(tmp_44, dim=-1)
        tmp_44 = None
        tmp_46 = torch.nn.functional.dropout(tmp_45, 0.1, False, False)
        tmp_45 = None
        tmp_47 = torch.matmul(tmp_46, tmp_29)
        tmp_29 = None
        tmp_48 = tmp_10[slice(None, 45, None), slice(None, 45, None), slice(None, None, None)]
        tmp_10 = None
        tmp_49 = tmp_46.permute(2, 0, 1, 3)
        tmp_46 = None
        tmp_50 = tmp_49.contiguous()
        tmp_49 = None
        tmp_51 = tmp_50.view(45, 4, 45)
        tmp_50 = None
        tmp_52 = torch.matmul(tmp_51, tmp_48)
        tmp_51 = tmp_48 = None
        tmp_53 = tmp_52.view(45, 1, 4, 8)
        tmp_52 = None
        tmp_54 = tmp_53.permute(1, 2, 0, 3)
        tmp_53 = None
        tmp_55 = tmp_47 + tmp_54
        tmp_47 = tmp_54 = None
        tmp_56 = tmp_55.permute(0, 2, 1, 3)
        tmp_55 = None
        tmp_57 = tmp_56.contiguous()
        tmp_56 = None
        tmp_58 = tmp_57.view((1, 45, 32))
        tmp_57 = None
        return (tmp_58, tmp_22, tmp_17)