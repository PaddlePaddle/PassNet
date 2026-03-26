import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18):
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
        tmp_12 = in_12
        tmp_13 = in_13
        tmp_14 = in_15.transpose(1, 2)
        tmp_15 = tmp_14.view(16, 1024, 16, 16)
        tmp_14 = None
        tmp_16 = torch.conv2d(tmp_15, tmp_11, tmp_10, (1, 1), (1, 1), (1, 1), 1024)
        tmp_15 = tmp_11 = tmp_10 = None
        tmp_17 = tmp_16.flatten(2)
        tmp_16 = None
        tmp_18 = tmp_17.transpose(1, 2)
        tmp_17 = None
        tmp_19 = torch.nn.functional.gelu(tmp_18)
        tmp_18 = None
        tmp_20 = torch.nn.functional.dropout(tmp_19, 0.0, False, False)
        tmp_19 = None
        tmp_21 = torch.nn.functional.linear(tmp_20, tmp_9, tmp_8)
        tmp_20 = tmp_9 = tmp_8 = None
        tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False)
        tmp_21 = None
        tmp_23 = tmp_22 + in_14
        tmp_22 = None
        tmp_24 = torch.nn.functional.layer_norm(tmp_23, (256,), tmp_13, tmp_12, 1e-05)
        tmp_23 = tmp_13 = tmp_12 = None
        tmp_25 = tmp_24.reshape(16, 16, 16, -1)
        tmp_24 = None
        tmp_26 = tmp_25.permute(0, 3, 1, 2)
        tmp_25 = None
        tmp_27 = tmp_26.contiguous()
        tmp_26 = None
        tmp_28 = in_16.flatten(2)
        tmp_29 = tmp_28.transpose(1, 2)
        tmp_28 = None
        tmp_30 = torch.nn.functional.linear(tmp_29, tmp_1, tmp_0)
        tmp_29 = tmp_1 = tmp_0 = None
        tmp_31 = tmp_30.permute(0, 2, 1)
        tmp_30 = None
        tmp_32 = tmp_31.reshape(16, -1, 128, 128)
        tmp_31 = None
        tmp_33 = torch.nn.functional.interpolate(tmp_32, size=(128, 128), mode='bilinear', align_corners=False)
        tmp_32 = None
        tmp_34 = in_17.flatten(2)
        tmp_35 = tmp_34.transpose(1, 2)
        tmp_34 = None
        tmp_36 = torch.nn.functional.linear(tmp_35, tmp_3, tmp_2)
        tmp_35 = tmp_3 = tmp_2 = None
        tmp_37 = tmp_36.permute(0, 2, 1)
        tmp_36 = None
        tmp_38 = tmp_37.reshape(16, -1, 64, 64)
        tmp_37 = None
        tmp_39 = torch.nn.functional.interpolate(tmp_38, size=(128, 128), mode='bilinear', align_corners=False)
        tmp_38 = None
        tmp_40 = in_18.flatten(2)
        tmp_41 = tmp_40.transpose(1, 2)
        tmp_40 = None
        tmp_42 = torch.nn.functional.linear(tmp_41, tmp_5, tmp_4)
        tmp_41 = tmp_5 = tmp_4 = None
        tmp_43 = tmp_42.permute(0, 2, 1)
        tmp_42 = None
        tmp_44 = tmp_43.reshape(16, -1, 32, 32)
        tmp_43 = None
        tmp_45 = torch.nn.functional.interpolate(tmp_44, size=(128, 128), mode='bilinear', align_corners=False)
        tmp_44 = None
        tmp_46 = tmp_27.flatten(2)
        tmp_27 = None
        tmp_47 = tmp_46.transpose(1, 2)
        tmp_46 = None
        tmp_48 = torch.nn.functional.linear(tmp_47, tmp_7, tmp_6)
        tmp_47 = tmp_7 = tmp_6 = None
        tmp_49 = tmp_48.permute(0, 2, 1)
        tmp_48 = None
        tmp_50 = tmp_49.reshape(16, -1, 16, 16)
        tmp_49 = None
        tmp_51 = torch.nn.functional.interpolate(tmp_50, size=(128, 128), mode='bilinear', align_corners=False)
        tmp_50 = None
        tmp_52 = torch.cat((tmp_51, tmp_45, tmp_39, tmp_33), dim=1)
        tmp_51 = tmp_45 = tmp_39 = tmp_33 = None
        return (tmp_52,)