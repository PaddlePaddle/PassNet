import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12):
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
        tmp_10 = in_10
        tmp_11 = in_11
        tmp_12 = torch.nn.functional.silu(in_12, inplace=True)
        tmp_13 = torch.conv2d(tmp_12, tmp_3, tmp_2, (1, 1), (3, 3), (1, 1), 1)
        tmp_12 = tmp_3 = tmp_2 = None
        tmp_14 = torch.flatten(tmp_13, 2)
        tmp_13 = None
        tmp_15 = torch.functional.norm(tmp_14, dim=-1, keepdim=True)
        tmp_16 = tmp_15 * 0.14433756729740643
        tmp_15 = None
        tmp_17 = tmp_16.clamp(min=1e-05)
        tmp_16 = None
        tmp_18 = tmp_14 / tmp_17
        tmp_14 = tmp_17 = None
        tmp_19 = tmp_18 * tmp_10
        tmp_18 = tmp_10 = None
        tmp_20 = torch.nn.functional.linear(tmp_19, tmp_11, None)
        tmp_19 = tmp_11 = None
        tmp_21 = torch.functional.norm(tmp_20, dim=-1, keepdim=True)
        tmp_22 = tmp_21 * 0.0625
        tmp_21 = None
        tmp_23 = tmp_22.clamp(min=1e-05)
        tmp_22 = None
        tmp_24 = tmp_20 / tmp_23
        tmp_23 = None
        tmp_25 = tmp_24 * tmp_4
        tmp_24 = tmp_4 = None
        tmp_26 = torch.nn.functional.linear(tmp_25, tmp_7, None)
        tmp_25 = tmp_7 = None
        tmp_27 = torch.nn.functional.silu(tmp_26, inplace=True)
        tmp_26 = None
        tmp_28 = torch.functional.split(tmp_27, [512, 512, 128], dim=2)
        tmp_27 = None
        tmp_29 = tmp_28[0]
        tmp_30 = tmp_28[1]
        tmp_31 = tmp_28[2]
        tmp_28 = None
        tmp_32 = tmp_31.unsqueeze(2)
        tmp_31 = None
        tmp_33 = tmp_9[None, None, slice(None, None, None)]
        tmp_9 = None
        tmp_34 = tmp_32 * tmp_33
        tmp_32 = tmp_33 = None
        tmp_35 = tmp_34 + tmp_8
        tmp_34 = tmp_8 = None
        tmp_36 = torch.unbind(tmp_35, dim=2)
        tmp_35 = None
        tmp_37 = tmp_36[0]
        tmp_38 = tmp_36[1]
        tmp_36 = None
        tmp_39 = tmp_38.permute(0, 2, 1)
        tmp_38 = None
        tmp_40 = torch.bmm(tmp_37, tmp_39)
        tmp_37 = tmp_39 = None
        tmp_41 = tmp_40 / 11.313708498984761
        tmp_40 = None
        tmp_42 = torch.nn.functional.relu(tmp_41)
        tmp_41 = None
        tmp_43 = torch.square(tmp_42)
        tmp_42 = None
        tmp_44 = torch.bmm(tmp_43, tmp_30)
        tmp_43 = tmp_30 = None
        tmp_45 = tmp_29 * tmp_44
        tmp_29 = tmp_44 = None
        tmp_46 = torch.nn.functional.linear(tmp_45, tmp_5, None)
        tmp_45 = tmp_5 = None
        tmp_47 = tmp_20 * tmp_6
        tmp_20 = tmp_6 = None
        tmp_48 = tmp_47 + tmp_46
        tmp_47 = tmp_46 = None
        tmp_49 = torch.nn.functional.linear(tmp_48, tmp_0, None)
        tmp_0 = None
        tmp_50 = torch.nn.functional.linear(tmp_48, tmp_1, None)
        tmp_48 = tmp_1 = None
        return (tmp_49, tmp_50)