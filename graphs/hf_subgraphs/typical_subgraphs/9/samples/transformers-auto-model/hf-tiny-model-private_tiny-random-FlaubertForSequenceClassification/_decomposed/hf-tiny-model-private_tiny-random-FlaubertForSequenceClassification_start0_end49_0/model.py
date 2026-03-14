import torch
from torch import device

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
        tmp_14 = tmp_1 != 2
        tmp_15 = tmp_14.sum(dim=1)
        tmp_14 = None
        tmp_16 = tmp_15.long()
        tmp_15 = None
        tmp_17 = tmp_16.max()
        tmp_16 = None
        tmp_18 = tmp_17.item()
        tmp_17 = None
        tmp_19 = tmp_18 <= 17
        tmp_18 = None
        tmp_20 = torch.ops.aten._assert_scalar.default(tmp_19, "Runtime assertion failed for expression u0 <= 17 on node 'le_1'")
        tmp_19 = tmp_20 = None
        tmp_21 = torch.arange(17, dtype=torch.int64, device=device(type='cuda', index=0))
        tmp_21 = None
        tmp_22 = tmp_2[slice(None, None, None), slice(None, 17, None)]
        tmp_2 = None
        tmp_23 = tmp_22.expand((1, 17))
        tmp_22 = None
        tmp_24 = tmp_1[slice(None, None, None), slice(-17, None, None)]
        tmp_1 = None
        tmp_25 = tmp_23[slice(None, None, None), slice(-17, None, None)]
        tmp_23 = None
        tmp_26 = tmp_0[slice(None, None, None), slice(-17, None, None)]
        tmp_27 = tmp_0[slice(None, None, None), slice(-17, None, None)]
        tmp_0 = None
        tmp_28 = torch.nn.functional.embedding(tmp_24, tmp_9, 2, None, 2.0, False, False)
        tmp_24 = None
        tmp_29 = torch.nn.functional.embedding(tmp_25, tmp_12, None, None, 2.0, False, False)
        tmp_25 = tmp_12 = None
        tmp_30 = tmp_29.expand_as(tmp_28)
        tmp_29 = None
        tmp_31 = tmp_28 + tmp_30
        tmp_28 = tmp_30 = None
        tmp_32 = torch.nn.functional.embedding(tmp_13, tmp_9, 2, None, 2.0, False, False)
        tmp_13 = tmp_9 = None
        tmp_33 = tmp_31 + tmp_32
        tmp_31 = tmp_32 = None
        tmp_34 = torch.nn.functional.layer_norm(tmp_33, (32,), tmp_11, tmp_10, 1e-12)
        tmp_33 = tmp_11 = tmp_10 = None
        tmp_35 = torch.nn.functional.dropout(tmp_34, p=0.1, training=False)
        tmp_34 = None
        tmp_36 = tmp_26.unsqueeze(-1)
        tmp_37 = tmp_36.to(torch.float32)
        tmp_36 = None
        tmp_35 *= tmp_37
        tmp_38 = tmp_35
        tmp_35 = tmp_37 = None
        tmp_39 = torch.nn.functional.linear(tmp_38, tmp_6, tmp_5)
        tmp_6 = tmp_5 = None
        tmp_40 = tmp_39.view(1, -1, 4, 8)
        tmp_39 = None
        tmp_41 = tmp_40.transpose(1, 2)
        tmp_40 = None
        tmp_42 = torch.nn.functional.linear(tmp_38, tmp_4, tmp_3)
        tmp_4 = tmp_3 = None
        tmp_43 = torch.nn.functional.linear(tmp_38, tmp_8, tmp_7)
        tmp_8 = tmp_7 = None
        tmp_44 = tmp_42.view(1, -1, 4, 8)
        tmp_42 = None
        tmp_45 = tmp_44.transpose(1, 2)
        tmp_44 = None
        tmp_46 = tmp_43.view(1, -1, 4, 8)
        tmp_43 = None
        tmp_47 = tmp_46.transpose(1, 2)
        tmp_46 = None
        tmp_48 = tmp_41 / 2.8284271247461903
        tmp_41 = None
        tmp_49 = tmp_45.transpose(2, 3)
        tmp_50 = torch.matmul(tmp_48, tmp_49)
        tmp_48 = tmp_49 = None
        tmp_51 = tmp_27.__eq__(0)
        tmp_52 = tmp_51.view((1, 1, 1, -1))
        tmp_51 = None
        tmp_53 = tmp_52.expand_as(tmp_50)
        tmp_52 = None
        tmp_54 = tmp_50.masked_fill_(tmp_53, -3.4028234663852886e+38)
        tmp_53 = tmp_54 = None
        tmp_55 = tmp_50.float()
        tmp_56 = torch.nn.functional.softmax(tmp_55, dim=-1)
        tmp_55 = None
        tmp_57 = tmp_56.type_as(tmp_50)
        tmp_56 = tmp_50 = None
        tmp_58 = torch.nn.functional.dropout(tmp_57, p=0.1, training=False)
        tmp_57 = None
        tmp_59 = torch.matmul(tmp_58, tmp_47)
        tmp_58 = None
        tmp_60 = tmp_59.transpose(1, 2)
        tmp_59 = None
        tmp_61 = tmp_60.contiguous()
        tmp_60 = None
        tmp_62 = tmp_61.view(1, -1, 32)
        tmp_61 = None
        return (tmp_27, tmp_62, tmp_45, tmp_26, tmp_38, tmp_47)