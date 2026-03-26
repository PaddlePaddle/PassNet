import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, in_2):
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
        tmp_21 = in_2
        tmp_22 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_23 = tmp_22.to(dtype=torch.float32)
        tmp_22 = None
        tmp_24 = 1.0 - tmp_23
        tmp_23 = None
        tmp_25 = tmp_24 * -3.4028234663852886e+38
        tmp_24 = None
        tmp_26 = tmp_2[slice(None, None, None), slice(None, 11, None)]
        tmp_2 = None
        tmp_27 = torch.nn.functional.embedding(tmp_1, tmp_7, 0, None, 2.0, False, False)
        tmp_1 = tmp_7 = None
        tmp_28 = torch.nn.functional.embedding(tmp_26, tmp_5, None, None, 2.0, False, False)
        tmp_26 = tmp_5 = None
        tmp_29 = torch.nn.functional.embedding(tmp_21, tmp_6, None, None, 2.0, False, False)
        tmp_21 = tmp_6 = None
        tmp_30 = tmp_27 + tmp_28
        tmp_27 = tmp_28 = None
        tmp_31 = tmp_30 + tmp_29
        tmp_30 = tmp_29 = None
        tmp_32 = torch.nn.functional.layer_norm(tmp_31, (768,), tmp_4, tmp_3, 1e-12)
        tmp_31 = tmp_4 = tmp_3 = None
        tmp_33 = torch.nn.functional.dropout(tmp_32, 0.1, False, False)
        tmp_32 = None
        tmp_34 = torch.nn.functional.linear(tmp_33, tmp_16, tmp_15)
        tmp_16 = tmp_15 = None
        tmp_35 = torch.nn.functional.linear(tmp_33, tmp_20, tmp_19)
        tmp_20 = tmp_19 = None
        tmp_36 = tmp_33.transpose(1, 2)
        tmp_37 = torch.conv1d(tmp_36, tmp_12, None, (1,), (4,), (1,), 768)
        tmp_36 = tmp_12 = None
        tmp_38 = torch.conv1d(tmp_37, tmp_13, None, (1,), (0,), (1,), 1)
        tmp_37 = tmp_13 = None
        tmp_38 += tmp_14
        tmp_39 = tmp_38
        tmp_38 = tmp_14 = None
        tmp_40 = tmp_39.transpose(1, 2)
        tmp_39 = None
        tmp_41 = torch.nn.functional.linear(tmp_33, tmp_18, tmp_17)
        tmp_18 = tmp_17 = None
        tmp_42 = tmp_41.view(1, -1, 6, 64)
        tmp_43 = tmp_42.transpose(1, 2)
        tmp_42 = None
        tmp_44 = tmp_34.view(1, -1, 6, 64)
        tmp_34 = None
        tmp_45 = tmp_44.transpose(1, 2)
        tmp_44 = None
        tmp_46 = tmp_35.view(1, -1, 6, 64)
        tmp_35 = None
        tmp_47 = tmp_46.transpose(1, 2)
        tmp_46 = None
        tmp_48 = torch.multiply(tmp_40, tmp_41)
        tmp_40 = tmp_41 = None
        tmp_49 = torch.nn.functional.linear(tmp_48, tmp_9, tmp_8)
        tmp_48 = tmp_9 = tmp_8 = None
        tmp_50 = torch.reshape(tmp_49, [-1, 9, 1])
        tmp_49 = None
        tmp_51 = torch.softmax(tmp_50, dim=1)
        tmp_50 = None
        tmp_52 = torch.nn.functional.linear(tmp_33, tmp_11, tmp_10)
        tmp_11 = tmp_10 = None
        tmp_53 = torch.reshape(tmp_52, [1, -1, 384])
        tmp_52 = None
        tmp_54 = tmp_53.transpose(1, 2)
        tmp_53 = None
        tmp_55 = tmp_54.contiguous()
        tmp_54 = None
        tmp_56 = tmp_55.unsqueeze(-1)
        tmp_55 = None
        tmp_57 = torch.nn.functional.unfold(tmp_56, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
        tmp_56 = None
        tmp_58 = tmp_57.transpose(1, 2)
        tmp_57 = None
        tmp_59 = tmp_58.reshape(1, -1, 384, 9)
        tmp_58 = None
        tmp_60 = torch.reshape(tmp_59, [-1, 64, 9])
        tmp_59 = None
        tmp_61 = torch.matmul(tmp_60, tmp_51)
        tmp_60 = tmp_51 = None
        tmp_62 = torch.reshape(tmp_61, [-1, 384])
        tmp_61 = None
        tmp_63 = tmp_45.transpose(-1, -2)
        tmp_45 = None
        tmp_64 = torch.matmul(tmp_43, tmp_63)
        tmp_43 = tmp_63 = None
        tmp_65 = tmp_64 / 8.0
        tmp_64 = None
        tmp_66 = tmp_65 + tmp_25
        tmp_65 = None
        tmp_67 = torch.nn.functional.softmax(tmp_66, dim=-1)
        tmp_66 = None
        tmp_68 = torch.nn.functional.dropout(tmp_67, 0.1, False, False)
        tmp_67 = None
        tmp_69 = torch.matmul(tmp_68, tmp_47)
        tmp_68 = tmp_47 = None
        tmp_70 = tmp_69.permute(0, 2, 1, 3)
        tmp_69 = None
        tmp_71 = tmp_70.contiguous()
        tmp_70 = None
        tmp_72 = torch.reshape(tmp_62, [1, -1, 6, 64])
        tmp_62 = None
        tmp_73 = torch.cat([tmp_71, tmp_72], 2)
        tmp_71 = tmp_72 = None
        tmp_74 = tmp_73.view(1, 11, 768)
        tmp_73 = None
        return (tmp_74, tmp_33, tmp_25)