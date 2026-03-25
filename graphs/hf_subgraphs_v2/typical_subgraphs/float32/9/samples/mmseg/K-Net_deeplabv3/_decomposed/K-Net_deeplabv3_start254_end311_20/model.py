import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, in_0, in_1, in_2, in_3):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = w_9
        tmp_10 = w_10
        tmp_11 = w_11
        tmp_12 = w_12
        tmp_13 = w_13
        tmp_14 = w_14
        tmp_15 = w_15
        tmp_16 = w_16
        tmp_17 = w_17
        tmp_18 = w_18
        tmp_19 = w_19
        tmp_20 = w_20
        tmp_21 = w_21
        tmp_22 = w_22
        tmp_23 = w_23
        tmp_24 = w_24
        tmp_25 = w_25
        tmp_26 = w_26
        tmp_27 = w_27
        tmp_28 = w_28
        tmp_29 = w_29
        tmp_30 = w_30
        tmp_31 = w_31
        tmp_32 = w_32
        tmp_33 = w_33
        tmp_34 = torch.nn.functional.layer_norm(in_0, (512,), tmp_3, tmp_2, 1e-05)
        tmp_3 = tmp_2 = None
        tmp_35 = torch.nn.functional.relu(tmp_34, inplace=True)
        tmp_34 = None
        tmp_36 = torch.nn.functional.linear(tmp_35, tmp_1, tmp_0)
        tmp_35 = tmp_1 = tmp_0 = None
        tmp_37 = tmp_36.permute(0, 1, 3, 2)
        tmp_36 = None
        tmp_38 = tmp_37.reshape(1, 150, 512, 1, 1)
        tmp_37 = None
        tmp_39 = in_3[slice(0, 1, None)]
        tmp_40 = tmp_38[0]
        tmp_38 = None
        tmp_41 = torch.conv2d(tmp_39, tmp_40, padding=0)
        tmp_39 = tmp_40 = None
        tmp_42 = torch.cat([tmp_41], dim=0)
        tmp_41 = None
        tmp_43 = tmp_42.reshape(1, 150, 64, 64)
        tmp_42 = None
        tmp_44 = in_1.permute(0, 1, 3, 2)
        tmp_45 = tmp_44.reshape(1, 150, 512, 1, 1)
        tmp_44 = None
        tmp_46 = torch.conv2d(in_2, tmp_11, tmp_10, (1, 1), (0, 0), (1, 1), 1)
        tmp_11 = tmp_10 = None
        tmp_47 = tmp_43.softmax(dim=1)
        tmp_43 = None
        tmp_48 = torch.functional.einsum('bnhw,bchw->bnc', tmp_47, tmp_46)
        tmp_47 = None
        tmp_49 = tmp_45.reshape(1, 150, 512, -1)
        tmp_45 = None
        tmp_50 = tmp_49.permute(0, 1, 3, 2)
        tmp_49 = None
        tmp_51 = tmp_48.reshape(-1, 256)
        tmp_48 = None
        tmp_52 = torch.nn.functional.linear(tmp_51, tmp_15, tmp_14)
        tmp_51 = tmp_15 = tmp_14 = None
        tmp_53 = tmp_52[slice(None, None, None), slice(None, 256, None)]
        tmp_54 = tmp_53.view(-1, 256)
        tmp_53 = None
        tmp_55 = tmp_52[slice(None, None, None), slice(-256, None, None)]
        tmp_52 = None
        tmp_56 = tmp_55.view(-1, 256)
        tmp_55 = None
        tmp_57 = tmp_50.reshape(300, -1, 256)
        tmp_50 = None
        tmp_58 = torch.nn.functional.linear(tmp_57, tmp_23, tmp_22)
        tmp_57 = tmp_23 = tmp_22 = None
        tmp_59 = tmp_58[Ellipsis, slice(None, 256, None)]
        tmp_60 = tmp_58[Ellipsis, slice(-256, None, None)]
        tmp_58 = None
        tmp_61 = tmp_54.unsqueeze(-2)
        tmp_54 = None
        tmp_62 = tmp_59 * tmp_61
        tmp_59 = tmp_61 = None
        tmp_63 = torch.nn.functional.linear(tmp_62, tmp_21, tmp_20)
        tmp_21 = tmp_20 = None
        tmp_64 = torch.nn.functional.layer_norm(tmp_63, (256,), tmp_25, tmp_24, 1e-05)
        tmp_63 = tmp_25 = tmp_24 = None
        tmp_65 = torch.nn.functional.linear(tmp_62, tmp_33, tmp_32)
        tmp_62 = tmp_33 = tmp_32 = None
        tmp_66 = torch.nn.functional.layer_norm(tmp_65, (256,), tmp_29, tmp_28, 1e-05)
        tmp_65 = tmp_29 = tmp_28 = None
        tmp_67 = tmp_64.sigmoid()
        tmp_64 = None
        tmp_68 = tmp_66.sigmoid()
        tmp_66 = None
        tmp_69 = torch.nn.functional.layer_norm(tmp_56, (256,), tmp_31, tmp_30, 1e-05)
        tmp_56 = tmp_31 = tmp_30 = None
        tmp_70 = torch.nn.functional.layer_norm(tmp_60, (256,), tmp_27, tmp_26, 1e-05)
        tmp_60 = tmp_27 = tmp_26 = None
        tmp_71 = tmp_69.unsqueeze(-2)
        tmp_69 = None
        tmp_72 = tmp_68 * tmp_71
        tmp_68 = tmp_71 = None
        tmp_73 = tmp_67 * tmp_70
        tmp_67 = tmp_70 = None
        tmp_74 = tmp_72 + tmp_73
        tmp_72 = tmp_73 = None
        tmp_75 = torch.nn.functional.linear(tmp_74, tmp_17, tmp_16)
        tmp_74 = tmp_17 = tmp_16 = None
        tmp_76 = torch.nn.functional.layer_norm(tmp_75, (256,), tmp_19, tmp_18, 1e-05)
        tmp_75 = tmp_19 = tmp_18 = None
        tmp_77 = torch.nn.functional.relu(tmp_76, inplace=True)
        tmp_76 = None
        tmp_78 = tmp_77.reshape(1, 150, -1)
        tmp_77 = None
        tmp_79 = tmp_78.permute(1, 0, 2)
        tmp_78 = None
        tmp_80 = torch.nn.functional.multi_head_attention_forward(tmp_79, tmp_79, tmp_79, 512, 8, tmp_7, tmp_6, None, None, False, 0.0, tmp_5, tmp_4, training=False, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False)
        tmp_7 = tmp_6 = tmp_5 = tmp_4 = None
        tmp_81 = tmp_80[0]
        tmp_80 = None
        tmp_82 = torch.nn.functional.dropout(tmp_81, 0.0, False, False)
        tmp_81 = None
        tmp_83 = torch.nn.functional.dropout(tmp_82, 0.0, False, False)
        tmp_82 = None
        tmp_84 = tmp_79 + tmp_83
        tmp_79 = tmp_83 = None
        tmp_85 = torch.nn.functional.layer_norm(tmp_84, (512,), tmp_9, tmp_8, 1e-05)
        tmp_84 = tmp_9 = tmp_8 = None
        tmp_86 = tmp_85.permute(1, 0, 2)
        tmp_85 = None
        tmp_87 = tmp_86.reshape(1, 150, -1, 512)
        tmp_86 = None
        tmp_88 = torch.nn.functional.linear(tmp_87, tmp_13, tmp_12)
        tmp_13 = tmp_12 = None
        tmp_89 = torch.nn.functional.relu(tmp_88, inplace=True)
        tmp_88 = None
        tmp_90 = torch.nn.functional.dropout(tmp_89, 0.0, False, False)
        tmp_89 = None
        return (tmp_90, tmp_87, tmp_46)