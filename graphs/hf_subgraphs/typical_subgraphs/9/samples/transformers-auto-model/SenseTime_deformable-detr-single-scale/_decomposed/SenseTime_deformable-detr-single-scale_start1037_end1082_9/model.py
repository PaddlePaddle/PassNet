import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1, in_2, in_3, in_4, in_5):
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
        tmp_10 = torch.nn.functional.relu(in_1, inplace=False)
        tmp_11 = torch.nn.functional.dropout(tmp_10, p=0.0, training=False)
        tmp_10 = None
        tmp_12 = torch.nn.functional.linear(tmp_11, tmp_1, tmp_0)
        tmp_11 = tmp_1 = tmp_0 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, p=0.1, training=False)
        tmp_12 = None
        tmp_14 = in_0 + tmp_13
        tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (256,), tmp_3, tmp_2, 1e-05)
        tmp_14 = tmp_3 = tmp_2 = None
        tmp_16 = tmp_15 + in_2
        tmp_17 = torch.nn.functional.linear(tmp_15, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_18 = in_3[Ellipsis, None]
        tmp_19 = ~tmp_18
        tmp_18 = None
        tmp_20 = tmp_17.masked_fill(tmp_19, 0.0)
        tmp_17 = tmp_19 = None
        tmp_21 = tmp_20.view(1, 625, 8, 32)
        tmp_20 = None
        tmp_22 = torch.nn.functional.linear(tmp_16, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_23 = tmp_22.view(1, 625, 8, 1, 4, 2)
        tmp_22 = None
        tmp_24 = torch.nn.functional.linear(tmp_16, tmp_5, tmp_4)
        tmp_16 = tmp_5 = tmp_4 = None
        tmp_25 = tmp_24.view(1, 625, 8, 4)
        tmp_24 = None
        tmp_26 = torch.nn.functional.softmax(tmp_25, -1)
        tmp_25 = None
        tmp_27 = tmp_26.view(1, 625, 8, 1, 4)
        tmp_26 = None
        tmp_28 = in_5[Ellipsis, 1]
        tmp_29 = in_5[Ellipsis, 0]
        tmp_30 = torch.stack([tmp_28, tmp_29], -1)
        tmp_28 = tmp_29 = None
        tmp_31 = in_4[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), None, slice(None, None, None)]
        tmp_32 = tmp_30[None, None, None, slice(None, None, None), None, slice(None, None, None)]
        tmp_30 = None
        tmp_33 = tmp_23 / tmp_32
        tmp_23 = tmp_32 = None
        tmp_34 = tmp_31 + tmp_33
        tmp_31 = tmp_33 = None
        tmp_35 = tmp_21.split([625], dim=1)
        tmp_21 = None
        tmp_36 = tmp_35[0]
        tmp_35 = None
        tmp_37 = 2 * tmp_34
        tmp_34 = None
        tmp_38 = tmp_37 - 1
        tmp_37 = None
        tmp_39 = tmp_36.flatten(2)
        tmp_36 = None
        tmp_40 = tmp_39.transpose(1, 2)
        tmp_39 = None
        tmp_41 = tmp_40.reshape(8, 32, 25, 25)
        tmp_40 = None
        tmp_42 = tmp_38[slice(None, None, None), slice(None, None, None), slice(None, None, None), 0]
        tmp_38 = None
        tmp_43 = tmp_42.transpose(1, 2)
        tmp_42 = None
        tmp_44 = tmp_43.flatten(0, 1)
        tmp_43 = None
        tmp_45 = torch.nn.functional.grid_sample(tmp_41, tmp_44, mode='bilinear', padding_mode='zeros', align_corners=False)
        tmp_41 = tmp_44 = None
        tmp_46 = tmp_27.transpose(1, 2)
        tmp_27 = None
        tmp_47 = tmp_46.reshape(8, 1, 625, 4)
        tmp_46 = None
        tmp_48 = torch.stack([tmp_45], dim=-2)
        tmp_45 = None
        tmp_49 = tmp_48.flatten(-2)
        tmp_48 = None
        tmp_50 = tmp_49 * tmp_47
        tmp_49 = tmp_47 = None
        tmp_51 = tmp_50.sum(-1)
        tmp_50 = None
        tmp_52 = tmp_51.view(1, 256, 625)
        tmp_51 = None
        tmp_53 = tmp_52.transpose(1, 2)
        tmp_52 = None
        tmp_54 = tmp_53.contiguous()
        tmp_53 = None
        return (tmp_15, tmp_54)