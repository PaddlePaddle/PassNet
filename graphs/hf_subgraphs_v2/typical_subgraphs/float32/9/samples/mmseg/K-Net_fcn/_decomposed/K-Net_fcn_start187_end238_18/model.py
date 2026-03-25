import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, in_0):
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
        tmp_32 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_33 = torch.nn.functional.dropout2d(tmp_32, 0.1, False, False)
        tmp_34 = torch.conv2d(tmp_33, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_33 = tmp_0 = None
        tmp_35 = tmp_1.clone()
        tmp_1 = None
        tmp_36 = tmp_35[None]
        tmp_35 = None
        tmp_37 = tmp_36.expand(1, 150, 512, 1, 1)
        tmp_36 = None
        tmp_38 = torch.conv2d(tmp_32, tmp_9, tmp_8, (1, 1), (0, 0), (1, 1), 1)
        tmp_9 = tmp_8 = None
        tmp_39 = tmp_34.softmax(dim=1)
        tmp_34 = None
        tmp_40 = torch.functional.einsum('bnhw,bchw->bnc', tmp_39, tmp_38)
        tmp_39 = None
        tmp_41 = tmp_37.reshape(1, 150, 512, -1)
        tmp_37 = None
        tmp_42 = tmp_41.permute(0, 1, 3, 2)
        tmp_41 = None
        tmp_43 = tmp_40.reshape(-1, 256)
        tmp_40 = None
        tmp_44 = torch.nn.functional.linear(tmp_43, tmp_13, tmp_12)
        tmp_43 = tmp_13 = tmp_12 = None
        tmp_45 = tmp_44[slice(None, None, None), slice(None, 256, None)]
        tmp_46 = tmp_45.view(-1, 256)
        tmp_45 = None
        tmp_47 = tmp_44[slice(None, None, None), slice(-256, None, None)]
        tmp_44 = None
        tmp_48 = tmp_47.view(-1, 256)
        tmp_47 = None
        tmp_49 = tmp_42.reshape(300, -1, 256)
        tmp_42 = None
        tmp_50 = torch.nn.functional.linear(tmp_49, tmp_21, tmp_20)
        tmp_49 = tmp_21 = tmp_20 = None
        tmp_51 = tmp_50[Ellipsis, slice(None, 256, None)]
        tmp_52 = tmp_50[Ellipsis, slice(-256, None, None)]
        tmp_50 = None
        tmp_53 = tmp_46.unsqueeze(-2)
        tmp_46 = None
        tmp_54 = tmp_51 * tmp_53
        tmp_51 = tmp_53 = None
        tmp_55 = torch.nn.functional.linear(tmp_54, tmp_19, tmp_18)
        tmp_19 = tmp_18 = None
        tmp_56 = torch.nn.functional.layer_norm(tmp_55, (256,), tmp_23, tmp_22, 1e-05)
        tmp_55 = tmp_23 = tmp_22 = None
        tmp_57 = torch.nn.functional.linear(tmp_54, tmp_31, tmp_30)
        tmp_54 = tmp_31 = tmp_30 = None
        tmp_58 = torch.nn.functional.layer_norm(tmp_57, (256,), tmp_27, tmp_26, 1e-05)
        tmp_57 = tmp_27 = tmp_26 = None
        tmp_59 = tmp_56.sigmoid()
        tmp_56 = None
        tmp_60 = tmp_58.sigmoid()
        tmp_58 = None
        tmp_61 = torch.nn.functional.layer_norm(tmp_48, (256,), tmp_29, tmp_28, 1e-05)
        tmp_48 = tmp_29 = tmp_28 = None
        tmp_62 = torch.nn.functional.layer_norm(tmp_52, (256,), tmp_25, tmp_24, 1e-05)
        tmp_52 = tmp_25 = tmp_24 = None
        tmp_63 = tmp_61.unsqueeze(-2)
        tmp_61 = None
        tmp_64 = tmp_60 * tmp_63
        tmp_60 = tmp_63 = None
        tmp_65 = tmp_59 * tmp_62
        tmp_59 = tmp_62 = None
        tmp_66 = tmp_64 + tmp_65
        tmp_64 = tmp_65 = None
        tmp_67 = torch.nn.functional.linear(tmp_66, tmp_15, tmp_14)
        tmp_66 = tmp_15 = tmp_14 = None
        tmp_68 = torch.nn.functional.layer_norm(tmp_67, (256,), tmp_17, tmp_16, 1e-05)
        tmp_67 = tmp_17 = tmp_16 = None
        tmp_69 = torch.nn.functional.relu(tmp_68, inplace=True)
        tmp_68 = None
        tmp_70 = tmp_69.reshape(1, 150, -1)
        tmp_69 = None
        tmp_71 = tmp_70.permute(1, 0, 2)
        tmp_70 = None
        tmp_72 = torch.nn.functional.multi_head_attention_forward(tmp_71, tmp_71, tmp_71, 512, 8, tmp_5, tmp_4, None, None, False, 0.0, tmp_3, tmp_2, training=False, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False)
        tmp_5 = tmp_4 = tmp_3 = tmp_2 = None
        tmp_73 = tmp_72[0]
        tmp_72 = None
        tmp_74 = torch.nn.functional.dropout(tmp_73, 0.0, False, False)
        tmp_73 = None
        tmp_75 = torch.nn.functional.dropout(tmp_74, 0.0, False, False)
        tmp_74 = None
        tmp_76 = tmp_71 + tmp_75
        tmp_71 = tmp_75 = None
        tmp_77 = torch.nn.functional.layer_norm(tmp_76, (512,), tmp_7, tmp_6, 1e-05)
        tmp_76 = tmp_7 = tmp_6 = None
        tmp_78 = tmp_77.permute(1, 0, 2)
        tmp_77 = None
        tmp_79 = tmp_78.reshape(1, 150, -1, 512)
        tmp_78 = None
        tmp_80 = torch.nn.functional.linear(tmp_79, tmp_11, tmp_10)
        tmp_11 = tmp_10 = None
        tmp_81 = torch.nn.functional.relu(tmp_80, inplace=True)
        tmp_80 = None
        tmp_82 = torch.nn.functional.dropout(tmp_81, 0.0, False, False)
        tmp_81 = None
        return (tmp_82, tmp_79, tmp_38, tmp_32)