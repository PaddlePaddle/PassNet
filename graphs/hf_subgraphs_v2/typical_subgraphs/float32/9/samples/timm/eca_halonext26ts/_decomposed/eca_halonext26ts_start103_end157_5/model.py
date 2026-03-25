import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = torch.nn.functional.silu(in_0, inplace=True)
        tmp_9 = torch.conv2d(tmp_8, tmp_7, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_7 = None
        tmp_10 = tmp_9.reshape(-1, 16, 2, 8, 2, 8)
        tmp_9 = None
        tmp_11 = tmp_10.permute(0, 1, 3, 5, 2, 4)
        tmp_10 = None
        tmp_12 = tmp_11.reshape(8, 16, -1, 4)
        tmp_11 = None
        tmp_13 = tmp_12.transpose(1, 3)
        tmp_12 = None
        tmp_14 = torch.conv2d(tmp_8, tmp_4, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_8 = tmp_4 = None
        tmp_15 = torch.nn.functional.pad(tmp_14, [2, 2, 2, 2], 'constant', None)
        tmp_14 = None
        tmp_16 = tmp_15.unfold(2, 12, 8)
        tmp_15 = None
        tmp_17 = tmp_16.unfold(3, 12, 8)
        tmp_16 = None
        tmp_18 = tmp_17.reshape(8, 48, 4, -1)
        tmp_17 = None
        tmp_19 = tmp_18.permute(0, 2, 3, 1)
        tmp_18 = None
        tmp_20 = torch.functional.split(tmp_19, [16, 32], dim=-1)
        tmp_19 = None
        tmp_21 = tmp_20[0]
        tmp_22 = tmp_20[1]
        tmp_20 = None
        tmp_23 = tmp_21.transpose(-1, -2)
        tmp_21 = None
        tmp_24 = tmp_13 @ tmp_23
        tmp_23 = None
        tmp_25 = tmp_24 * 0.25
        tmp_24 = None
        tmp_26 = tmp_13.reshape(-1, 8, 8, 16)
        tmp_13 = None
        tmp_27 = tmp_6.transpose(-1, -2)
        tmp_6 = None
        tmp_28 = tmp_26 @ tmp_27
        tmp_27 = None
        tmp_29 = tmp_28.reshape(-1, 8, 23)
        tmp_28 = None
        tmp_30 = torch.nn.functional.pad(tmp_29, [0, 1], 'constant', None)
        tmp_29 = None
        tmp_31 = tmp_30.flatten(1)
        tmp_30 = None
        tmp_32 = torch.nn.functional.pad(tmp_31, [0, 15], 'constant', None)
        tmp_31 = None
        tmp_33 = tmp_32.reshape(-1, 9, 23)
        tmp_32 = None
        tmp_34 = tmp_33[slice(None, None, None), slice(None, 8, None), slice(11, None, None)]
        tmp_33 = None
        tmp_35 = tmp_34.reshape(32, 8, 1, 8, 12)
        tmp_34 = None
        tmp_36 = tmp_35.expand(-1, -1, 12, -1, -1)
        tmp_35 = None
        tmp_37 = tmp_36.permute((0, 1, 3, 2, 4))
        tmp_36 = None
        tmp_38 = tmp_26.transpose(1, 2)
        tmp_26 = None
        tmp_39 = tmp_5.transpose(-1, -2)
        tmp_5 = None
        tmp_40 = tmp_38 @ tmp_39
        tmp_38 = tmp_39 = None
        tmp_41 = tmp_40.reshape(-1, 8, 23)
        tmp_40 = None
        tmp_42 = torch.nn.functional.pad(tmp_41, [0, 1], 'constant', None)
        tmp_41 = None
        tmp_43 = tmp_42.flatten(1)
        tmp_42 = None
        tmp_44 = torch.nn.functional.pad(tmp_43, [0, 15], 'constant', None)
        tmp_43 = None
        tmp_45 = tmp_44.reshape(-1, 9, 23)
        tmp_44 = None
        tmp_46 = tmp_45[slice(None, None, None), slice(None, 8, None), slice(11, None, None)]
        tmp_45 = None
        tmp_47 = tmp_46.reshape(32, 8, 1, 8, 12)
        tmp_46 = None
        tmp_48 = tmp_47.expand(-1, -1, 12, -1, -1)
        tmp_47 = None
        tmp_49 = tmp_48.permute((0, 3, 1, 4, 2))
        tmp_48 = None
        tmp_50 = tmp_49 + tmp_37
        tmp_49 = tmp_37 = None
        tmp_51 = tmp_50.reshape(8, 4, 64, -1)
        tmp_50 = None
        tmp_52 = tmp_25 + tmp_51
        tmp_25 = tmp_51 = None
        tmp_53 = tmp_52.softmax(dim=-1)
        tmp_52 = None
        tmp_54 = tmp_53 @ tmp_22
        tmp_53 = tmp_22 = None
        tmp_55 = tmp_54.transpose(1, 3)
        tmp_54 = None
        tmp_56 = tmp_55.reshape(-1, 8, 8, 2, 2)
        tmp_55 = None
        tmp_57 = tmp_56.permute(0, 3, 1, 4, 2)
        tmp_56 = None
        tmp_58 = tmp_57.contiguous()
        tmp_57 = None
        tmp_59 = tmp_58.view(1, 256, 16, 16)
        tmp_58 = None
        tmp_60 = torch.nn.functional.batch_norm(tmp_59, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_59 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_61 = torch.nn.functional.silu(tmp_60, inplace=True)
        tmp_60 = None
        return (tmp_61,)