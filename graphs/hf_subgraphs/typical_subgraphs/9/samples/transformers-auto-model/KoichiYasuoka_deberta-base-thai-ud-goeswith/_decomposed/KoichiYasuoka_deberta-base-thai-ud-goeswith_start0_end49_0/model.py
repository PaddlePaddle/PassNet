import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10):
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
        tmp_13 = tmp_2[slice(None, None, None), slice(None, 27, None)]
        tmp_2 = None
        tmp_14 = torch.nn.functional.embedding(tmp_1, tmp_6, 0, None, 2.0, False, False)
        tmp_1 = tmp_6 = None
        tmp_15 = tmp_13.long()
        tmp_13 = None
        tmp_16 = torch.nn.functional.embedding(tmp_15, tmp_5, None, None, 2.0, False, False)
        tmp_15 = tmp_5 = None
        tmp_17 = tmp_14 + tmp_16
        tmp_14 = tmp_16 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (768,), tmp_4, tmp_3, 1e-07)
        tmp_17 = tmp_4 = tmp_3 = None
        tmp_19 = tmp_0.unsqueeze(2)
        tmp_20 = tmp_19.to(torch.float32)
        tmp_19 = None
        tmp_21 = tmp_18 * tmp_20
        tmp_18 = tmp_20 = None
        tmp_22 = torch.nn.functional.dropout(tmp_21, 0.1, False, False)
        tmp_21 = None
        tmp_23 = tmp_0.unsqueeze(1)
        tmp_0 = None
        tmp_24 = tmp_23.unsqueeze(2)
        tmp_23 = None
        tmp_25 = tmp_24.squeeze(-2)
        tmp_26 = tmp_25.unsqueeze(-1)
        tmp_25 = None
        tmp_27 = tmp_24 * tmp_26
        tmp_24 = tmp_26 = None
        tmp_28 = torch.nn.functional.linear(tmp_22, tmp_10, tmp_9)
        tmp_10 = tmp_9 = None
        tmp_29 = tmp_28.view((1, 27, 12, -1))
        tmp_28 = None
        tmp_30 = tmp_29.permute(0, 2, 1, 3)
        tmp_29 = None
        tmp_31 = tmp_30.contiguous()
        tmp_30 = None
        tmp_32 = tmp_31.view(-1, 27, 64)
        tmp_31 = None
        tmp_33 = torch.nn.functional.linear(tmp_22, tmp_8, tmp_7)
        tmp_8 = tmp_7 = None
        tmp_34 = tmp_33.view((1, 27, 12, -1))
        tmp_33 = None
        tmp_35 = tmp_34.permute(0, 2, 1, 3)
        tmp_34 = None
        tmp_36 = tmp_35.contiguous()
        tmp_35 = None
        tmp_37 = tmp_36.view(-1, 27, 64)
        tmp_36 = None
        tmp_38 = torch.nn.functional.linear(tmp_22, tmp_12, tmp_11)
        tmp_12 = tmp_11 = None
        tmp_39 = tmp_38.view((1, 27, 12, -1))
        tmp_38 = None
        tmp_40 = tmp_39.permute(0, 2, 1, 3)
        tmp_39 = None
        tmp_41 = tmp_40.contiguous()
        tmp_40 = None
        tmp_42 = tmp_41.view(-1, 27, 64)
        tmp_41 = None
        tmp_43 = torch.tensor(64, dtype=torch.float32)
        tmp_44 = tmp_43 * 1
        tmp_43 = None
        tmp_45 = torch.sqrt(tmp_44)
        tmp_44 = None
        tmp_46 = tmp_37.transpose(-1, -2)
        tmp_37 = None
        tmp_47 = tmp_45.to(dtype=torch.float32)
        tmp_45 = None
        tmp_48 = tmp_46 / tmp_47
        tmp_46 = tmp_47 = None
        tmp_49 = torch.bmm(tmp_32, tmp_48)
        tmp_32 = tmp_48 = None
        tmp_50 = tmp_49.view(-1, 12, 27, 27)
        tmp_49 = None
        tmp_51 = tmp_27.bool()
        tmp_52 = ~tmp_51
        tmp_51 = None
        tmp_53 = tmp_50.masked_fill(tmp_52, -3.4028234663852886e+38)
        tmp_50 = tmp_52 = None
        tmp_54 = torch.nn.functional.softmax(tmp_53, dim=-1)
        tmp_53 = None
        tmp_55 = torch.nn.functional.dropout(tmp_54, 0.1, False, False)
        tmp_54 = None
        tmp_56 = tmp_55.view(-1, 27, 27)
        tmp_55 = None
        tmp_57 = torch.bmm(tmp_56, tmp_42)
        tmp_56 = tmp_42 = None
        tmp_58 = tmp_57.view(-1, 12, 27, 64)
        tmp_57 = None
        tmp_59 = tmp_58.permute(0, 2, 1, 3)
        tmp_58 = None
        tmp_60 = tmp_59.contiguous()
        tmp_59 = None
        tmp_61 = tmp_60.view((1, 27, -1))
        tmp_60 = None
        return (tmp_27, tmp_61, tmp_22)