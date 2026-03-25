import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
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
        tmp_10 = torch.nn.functional.relu(in_11, inplace=True)
        tmp_11 = torch.flatten(tmp_10, 2)
        tmp_10 = None
        tmp_12 = torch.functional.norm(tmp_11, dim=-1, keepdim=True)
        tmp_13 = tmp_12 * 0.07216878364870322
        tmp_12 = None
        tmp_14 = tmp_13.clamp(min=1e-05)
        tmp_13 = None
        tmp_15 = tmp_11 / tmp_14
        tmp_11 = tmp_14 = None
        tmp_16 = tmp_15 * tmp_8
        tmp_15 = tmp_8 = None
        tmp_17 = torch.nn.functional.linear(tmp_16, tmp_9, None)
        tmp_16 = tmp_9 = None
        tmp_18 = torch.cat([in_10, tmp_17], dim=2)
        tmp_17 = None
        tmp_19 = torch.functional.norm(tmp_18, dim=-1, keepdim=True)
        tmp_20 = tmp_19 * 0.0625
        tmp_19 = None
        tmp_21 = tmp_20.clamp(min=1e-05)
        tmp_20 = None
        tmp_22 = tmp_18 / tmp_21
        tmp_21 = None
        tmp_23 = tmp_22 * tmp_2
        tmp_22 = tmp_2 = None
        tmp_24 = torch.nn.functional.linear(tmp_23, tmp_5, None)
        tmp_23 = tmp_5 = None
        tmp_25 = torch.nn.functional.silu(tmp_24, inplace=True)
        tmp_24 = None
        tmp_26 = torch.functional.split(tmp_25, [512, 512, 128], dim=2)
        tmp_25 = None
        tmp_27 = tmp_26[0]
        tmp_28 = tmp_26[1]
        tmp_29 = tmp_26[2]
        tmp_26 = None
        tmp_30 = tmp_29.unsqueeze(2)
        tmp_29 = None
        tmp_31 = tmp_7[None, None, slice(None, None, None)]
        tmp_7 = None
        tmp_32 = tmp_30 * tmp_31
        tmp_30 = tmp_31 = None
        tmp_33 = tmp_32 + tmp_6
        tmp_32 = tmp_6 = None
        tmp_34 = torch.unbind(tmp_33, dim=2)
        tmp_33 = None
        tmp_35 = tmp_34[0]
        tmp_36 = tmp_34[1]
        tmp_34 = None
        tmp_37 = tmp_36.permute(0, 2, 1)
        tmp_36 = None
        tmp_38 = torch.bmm(tmp_35, tmp_37)
        tmp_35 = tmp_37 = None
        tmp_39 = tmp_38 / 11.313708498984761
        tmp_38 = None
        tmp_40 = torch.nn.functional.relu(tmp_39)
        tmp_39 = None
        tmp_41 = torch.square(tmp_40)
        tmp_40 = None
        tmp_42 = torch.bmm(tmp_41, tmp_28)
        tmp_41 = tmp_28 = None
        tmp_43 = tmp_27 * tmp_42
        tmp_27 = tmp_42 = None
        tmp_44 = torch.nn.functional.linear(tmp_43, tmp_3, None)
        tmp_43 = tmp_3 = None
        tmp_45 = tmp_18 * tmp_4
        tmp_18 = tmp_4 = None
        tmp_46 = tmp_45 + tmp_44
        tmp_45 = tmp_44 = None
        tmp_47 = torch.nn.functional.linear(tmp_46, tmp_0, None)
        tmp_0 = None
        tmp_48 = torch.nn.functional.linear(tmp_46, tmp_1, None)
        tmp_46 = tmp_1 = None
        return (tmp_47, tmp_48)