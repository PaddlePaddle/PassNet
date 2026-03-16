import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, in_2):
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
        tmp_13 = w_11
        tmp_14 = w_12
        tmp_15 = w_13
        tmp_16 = w_14
        tmp_17 = w_15
        tmp_18 = w_16
        tmp_19 = w_17
        tmp_20 = w_18
        tmp_21 = w_19
        tmp_22 = w_20
        tmp_23 = in_2
        tmp_24 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_25 = tmp_24.to(dtype=torch.float32)
        tmp_24 = None
        tmp_26 = 1.0 - tmp_25
        tmp_25 = None
        tmp_27 = tmp_26 * -3.4028234663852886e+38
        tmp_26 = None
        tmp_28 = tmp_2[slice(None, None, None), slice(None, 45, None)]
        tmp_2 = None
        tmp_29 = torch.nn.functional.embedding(tmp_1, tmp_7, 0, None, 2.0, False, False)
        tmp_1 = tmp_7 = None
        tmp_30 = torch.nn.functional.embedding(tmp_28, tmp_5, None, None, 2.0, False, False)
        tmp_28 = tmp_5 = None
        tmp_31 = torch.nn.functional.embedding(tmp_23, tmp_6, None, None, 2.0, False, False)
        tmp_23 = tmp_6 = None
        tmp_32 = tmp_29 + tmp_30
        tmp_29 = tmp_30 = None
        tmp_33 = tmp_32 + tmp_31
        tmp_32 = tmp_31 = None
        tmp_34 = torch.nn.functional.layer_norm(tmp_33, (768,), tmp_4, tmp_3, 1e-12)
        tmp_33 = tmp_4 = tmp_3 = None
        tmp_35 = torch.nn.functional.dropout(tmp_34, 0.1, False, False)
        tmp_34 = None
        tmp_36 = torch.nn.functional.linear(tmp_35, tmp_9, tmp_8)
        tmp_35 = tmp_9 = tmp_8 = None
        tmp_37 = torch.nn.functional.linear(tmp_36, tmp_18, tmp_17)
        tmp_18 = tmp_17 = None
        tmp_38 = torch.nn.functional.linear(tmp_36, tmp_22, tmp_21)
        tmp_22 = tmp_21 = None
        tmp_39 = tmp_36.transpose(1, 2)
        tmp_40 = torch.conv1d(tmp_39, tmp_14, None, (1,), (4,), (1,), 32)
        tmp_39 = tmp_14 = None
        tmp_41 = torch.conv1d(tmp_40, tmp_15, None, (1,), (0,), (1,), 1)
        tmp_40 = tmp_15 = None
        tmp_41 += tmp_16
        tmp_42 = tmp_41
        tmp_41 = tmp_16 = None
        tmp_43 = tmp_42.transpose(1, 2)
        tmp_42 = None
        tmp_44 = torch.nn.functional.linear(tmp_36, tmp_20, tmp_19)
        tmp_20 = tmp_19 = None
        tmp_45 = tmp_44.view(1, -1, 2, 8)
        tmp_46 = tmp_45.transpose(1, 2)
        tmp_45 = None
        tmp_47 = tmp_37.view(1, -1, 2, 8)
        tmp_37 = None
        tmp_48 = tmp_47.transpose(1, 2)
        tmp_47 = None
        tmp_49 = tmp_38.view(1, -1, 2, 8)
        tmp_38 = None
        tmp_50 = tmp_49.transpose(1, 2)
        tmp_49 = None
        tmp_51 = torch.multiply(tmp_43, tmp_44)
        tmp_43 = tmp_44 = None
        tmp_52 = torch.nn.functional.linear(tmp_51, tmp_11, tmp_10)
        tmp_51 = tmp_11 = tmp_10 = None
        tmp_53 = torch.reshape(tmp_52, [-1, 9, 1])
        tmp_52 = None
        tmp_54 = torch.softmax(tmp_53, dim=1)
        tmp_53 = None
        tmp_55 = torch.nn.functional.linear(tmp_36, tmp_13, tmp_12)
        tmp_13 = tmp_12 = None
        tmp_56 = torch.reshape(tmp_55, [1, -1, 16])
        tmp_55 = None
        tmp_57 = tmp_56.transpose(1, 2)
        tmp_56 = None
        tmp_58 = tmp_57.contiguous()
        tmp_57 = None
        tmp_59 = tmp_58.unsqueeze(-1)
        tmp_58 = None
        tmp_60 = torch.nn.functional.unfold(tmp_59, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
        tmp_59 = None
        tmp_61 = tmp_60.transpose(1, 2)
        tmp_60 = None
        tmp_62 = tmp_61.reshape(1, -1, 16, 9)
        tmp_61 = None
        tmp_63 = torch.reshape(tmp_62, [-1, 8, 9])
        tmp_62 = None
        tmp_64 = torch.matmul(tmp_63, tmp_54)
        tmp_63 = tmp_54 = None
        tmp_65 = torch.reshape(tmp_64, [-1, 16])
        tmp_64 = None
        tmp_66 = tmp_48.transpose(-1, -2)
        tmp_48 = None
        tmp_67 = torch.matmul(tmp_46, tmp_66)
        tmp_46 = tmp_66 = None
        tmp_68 = tmp_67 / 2.8284271247461903
        tmp_67 = None
        tmp_69 = tmp_68 + tmp_27
        tmp_68 = None
        tmp_70 = torch.nn.functional.softmax(tmp_69, dim=-1)
        tmp_69 = None
        tmp_71 = torch.nn.functional.dropout(tmp_70, 0.1, False, False)
        tmp_70 = None
        tmp_72 = torch.matmul(tmp_71, tmp_50)
        tmp_71 = tmp_50 = None
        tmp_73 = tmp_72.permute(0, 2, 1, 3)
        tmp_72 = None
        tmp_74 = tmp_73.contiguous()
        tmp_73 = None
        tmp_75 = torch.reshape(tmp_65, [1, -1, 2, 8])
        tmp_65 = None
        tmp_76 = torch.cat([tmp_74, tmp_75], 2)
        tmp_74 = tmp_75 = None
        tmp_77 = tmp_76.view(1, 45, 32)
        tmp_76 = None
        return (tmp_77, tmp_27, tmp_36)