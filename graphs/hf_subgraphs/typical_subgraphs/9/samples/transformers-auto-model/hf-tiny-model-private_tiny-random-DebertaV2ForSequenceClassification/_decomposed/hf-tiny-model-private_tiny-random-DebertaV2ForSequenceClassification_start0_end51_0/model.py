import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_2):
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
        tmp_14 = in_2
        tmp_15 = tmp_2[slice(None, None, None), slice(None, 11, None)]
        tmp_2 = None
        tmp_16 = torch.nn.functional.embedding(tmp_1, tmp_7, 0, None, 2.0, False, False)
        tmp_1 = tmp_7 = None
        tmp_17 = tmp_15.long()
        tmp_15 = None
        tmp_18 = torch.nn.functional.embedding(tmp_17, tmp_5, None, None, 2.0, False, False)
        tmp_17 = tmp_5 = None
        tmp_19 = tmp_16 + tmp_18
        tmp_16 = tmp_18 = None
        tmp_20 = torch.nn.functional.embedding(tmp_14, tmp_6, None, None, 2.0, False, False)
        tmp_14 = tmp_6 = None
        tmp_21 = tmp_19 + tmp_20
        tmp_19 = tmp_20 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (32,), tmp_4, tmp_3, 1e-07)
        tmp_21 = tmp_4 = tmp_3 = None
        tmp_23 = tmp_0.unsqueeze(2)
        tmp_24 = tmp_23.to(torch.float32)
        tmp_23 = None
        tmp_25 = tmp_22 * tmp_24
        tmp_22 = tmp_24 = None
        tmp_26 = torch.nn.functional.dropout(tmp_25, 0.1, False, False)
        tmp_25 = None
        tmp_27 = tmp_0.unsqueeze(1)
        tmp_0 = None
        tmp_28 = tmp_27.unsqueeze(2)
        tmp_27 = None
        tmp_29 = tmp_28.squeeze(-2)
        tmp_30 = tmp_29.unsqueeze(-1)
        tmp_29 = None
        tmp_31 = tmp_28 * tmp_30
        tmp_28 = tmp_30 = None
        tmp_32 = torch.nn.functional.linear(tmp_26, tmp_11, tmp_10)
        tmp_11 = tmp_10 = None
        tmp_33 = tmp_32.view((1, 11, 4, -1))
        tmp_32 = None
        tmp_34 = tmp_33.permute(0, 2, 1, 3)
        tmp_33 = None
        tmp_35 = tmp_34.contiguous()
        tmp_34 = None
        tmp_36 = tmp_35.view(-1, 11, 8)
        tmp_35 = None
        tmp_37 = torch.nn.functional.linear(tmp_26, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_38 = tmp_37.view((1, 11, 4, -1))
        tmp_37 = None
        tmp_39 = tmp_38.permute(0, 2, 1, 3)
        tmp_38 = None
        tmp_40 = tmp_39.contiguous()
        tmp_39 = None
        tmp_41 = tmp_40.view(-1, 11, 8)
        tmp_40 = None
        tmp_42 = torch.nn.functional.linear(tmp_26, tmp_13, tmp_12)
        tmp_13 = tmp_12 = None
        tmp_43 = tmp_42.view((1, 11, 4, -1))
        tmp_42 = None
        tmp_44 = tmp_43.permute(0, 2, 1, 3)
        tmp_43 = None
        tmp_45 = tmp_44.contiguous()
        tmp_44 = None
        tmp_46 = tmp_45.view(-1, 11, 8)
        tmp_45 = None
        tmp_47 = torch.tensor(8, dtype=torch.float32)
        tmp_48 = tmp_47 * 1
        tmp_47 = None
        tmp_49 = torch.sqrt(tmp_48)
        tmp_48 = None
        tmp_50 = tmp_41.transpose(-1, -2)
        tmp_41 = None
        tmp_51 = tmp_49.to(dtype=torch.float32)
        tmp_49 = None
        tmp_52 = tmp_50 / tmp_51
        tmp_50 = tmp_51 = None
        tmp_53 = torch.bmm(tmp_36, tmp_52)
        tmp_36 = tmp_52 = None
        tmp_54 = tmp_53.view(-1, 4, 11, 11)
        tmp_53 = None
        tmp_55 = tmp_31.bool()
        tmp_56 = ~tmp_55
        tmp_55 = None
        tmp_57 = tmp_54.masked_fill(tmp_56, -3.4028234663852886e+38)
        tmp_54 = tmp_56 = None
        tmp_58 = torch.nn.functional.softmax(tmp_57, dim=-1)
        tmp_57 = None
        tmp_59 = torch.nn.functional.dropout(tmp_58, 0.1, False, False)
        tmp_58 = None
        tmp_60 = tmp_59.view(-1, 11, 11)
        tmp_59 = None
        tmp_61 = torch.bmm(tmp_60, tmp_46)
        tmp_60 = tmp_46 = None
        tmp_62 = tmp_61.view(-1, 4, 11, 8)
        tmp_61 = None
        tmp_63 = tmp_62.permute(0, 2, 1, 3)
        tmp_62 = None
        tmp_64 = tmp_63.contiguous()
        tmp_63 = None
        tmp_65 = tmp_64.view((1, 11, -1))
        tmp_64 = None
        return (tmp_31, tmp_65, tmp_26)