import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = torch.nn.functional.silu(in_0, inplace=True)
        tmp_8 = torch.conv2d(tmp_7, tmp_6, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_7 = tmp_6 = None
        tmp_9 = torch.functional.split(tmp_8, [64, 64, 512], dim=1)
        tmp_8 = None
        tmp_10 = tmp_9[0]
        tmp_11 = tmp_9[1]
        tmp_12 = tmp_9[2]
        tmp_9 = None
        tmp_13 = tmp_10.reshape(4, 16, -1)
        tmp_10 = None
        tmp_14 = tmp_13.transpose(-1, -2)
        tmp_13 = None
        tmp_15 = tmp_11.reshape(4, 16, -1)
        tmp_11 = None
        tmp_16 = tmp_12.reshape(4, 128, -1)
        tmp_12 = None
        tmp_17 = tmp_16.transpose(-1, -2)
        tmp_16 = None
        tmp_18 = tmp_14 @ tmp_15
        tmp_15 = None
        tmp_19 = tmp_18 * 0.25
        tmp_18 = None
        tmp_20 = tmp_14.reshape(4, 16, 16, -1)
        tmp_14 = None
        tmp_21 = tmp_5.transpose(-1, -2)
        tmp_5 = None
        tmp_22 = tmp_20 @ tmp_21
        tmp_21 = None
        tmp_23 = tmp_22.reshape(-1, 16, 31)
        tmp_22 = None
        tmp_24 = torch.nn.functional.pad(tmp_23, [0, 1], 'constant', None)
        tmp_23 = None
        tmp_25 = tmp_24.flatten(1)
        tmp_24 = None
        tmp_26 = torch.nn.functional.pad(tmp_25, [0, 15], 'constant', None)
        tmp_25 = None
        tmp_27 = tmp_26.reshape(-1, 17, 31)
        tmp_26 = None
        tmp_28 = tmp_27[slice(None, None, None), slice(None, 16, None), slice(15, None, None)]
        tmp_27 = None
        tmp_29 = tmp_28.reshape(4, 16, 1, 16, 16)
        tmp_28 = None
        tmp_30 = tmp_29.expand(-1, -1, 16, -1, -1)
        tmp_29 = None
        tmp_31 = tmp_30.permute((0, 1, 3, 2, 4))
        tmp_30 = None
        tmp_32 = tmp_20.transpose(1, 2)
        tmp_20 = None
        tmp_33 = tmp_4.transpose(-1, -2)
        tmp_4 = None
        tmp_34 = tmp_32 @ tmp_33
        tmp_32 = tmp_33 = None
        tmp_35 = tmp_34.reshape(-1, 16, 31)
        tmp_34 = None
        tmp_36 = torch.nn.functional.pad(tmp_35, [0, 1], 'constant', None)
        tmp_35 = None
        tmp_37 = tmp_36.flatten(1)
        tmp_36 = None
        tmp_38 = torch.nn.functional.pad(tmp_37, [0, 15], 'constant', None)
        tmp_37 = None
        tmp_39 = tmp_38.reshape(-1, 17, 31)
        tmp_38 = None
        tmp_40 = tmp_39[slice(None, None, None), slice(None, 16, None), slice(15, None, None)]
        tmp_39 = None
        tmp_41 = tmp_40.reshape(4, 16, 1, 16, 16)
        tmp_40 = None
        tmp_42 = tmp_41.expand(-1, -1, 16, -1, -1)
        tmp_41 = None
        tmp_43 = tmp_42.permute((0, 3, 1, 4, 2))
        tmp_42 = None
        tmp_44 = tmp_43 + tmp_31
        tmp_43 = tmp_31 = None
        tmp_45 = tmp_44.reshape(4, 256, 256)
        tmp_44 = None
        tmp_46 = tmp_19 + tmp_45
        tmp_19 = tmp_45 = None
        tmp_47 = tmp_46.softmax(dim=-1)
        tmp_46 = None
        tmp_48 = tmp_47 @ tmp_17
        tmp_47 = tmp_17 = None
        tmp_49 = tmp_48.transpose(-1, -2)
        tmp_48 = None
        tmp_50 = tmp_49.reshape(1, 512, 16, 16)
        tmp_49 = None
        tmp_51 = torch.nn.functional.avg_pool2d(tmp_50, 2, 2, 0, False, True, None)
        tmp_50 = None
        tmp_52 = torch.nn.functional.batch_norm(tmp_51, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_51 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_53 = torch.nn.functional.silu(tmp_52, inplace=True)
        tmp_52 = None
        return (tmp_53,)