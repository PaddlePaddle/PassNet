import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.nn.functional.silu(in_0, inplace=True)
        tmp_5 = torch.conv2d(tmp_4, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_4 = tmp_1 = tmp_0 = None
        tmp_6 = torch.cat((in_1, tmp_5), 1)
        tmp_5 = None
        tmp_7 = in_2.view(1, 144, -1)
        tmp_8 = tmp_6.view(1, 144, -1)
        tmp_9 = torch.cat([tmp_7, tmp_8], 2)
        tmp_7 = tmp_8 = None
        tmp_10 = tmp_3[0]
        tmp_11 = tmp_3[1]
        tmp_3 = None
        tmp_12 = torch.arange(end=40, device=device(type='cuda', index=0), dtype=torch.float32)
        tmp_13 = tmp_12 + 0.5
        tmp_12 = None
        tmp_14 = torch.arange(end=40, device=device(type='cuda', index=0), dtype=torch.float32)
        tmp_15 = tmp_14 + 0.5
        tmp_14 = None
        tmp_16 = torch.functional.meshgrid(tmp_15, tmp_13, indexing='ij')
        tmp_15 = tmp_13 = None
        tmp_17 = tmp_16[0]
        tmp_18 = tmp_16[1]
        tmp_16 = None
        tmp_19 = torch.stack((tmp_18, tmp_17), -1)
        tmp_18 = tmp_17 = None
        tmp_20 = tmp_19.view(-1, 2)
        tmp_19 = None
        tmp_21 = torch.ops.aten._local_scalar_dense(tmp_10)
        tmp_10 = None
        tmp_22 = torch.full((1600, 1), tmp_21, dtype=torch.float32, device=device(type='cuda', index=0))
        tmp_21 = None
        tmp_23 = torch.arange(end=20, device=device(type='cuda', index=0), dtype=torch.float32)
        tmp_24 = tmp_23 + 0.5
        tmp_23 = None
        tmp_25 = torch.arange(end=20, device=device(type='cuda', index=0), dtype=torch.float32)
        tmp_26 = tmp_25 + 0.5
        tmp_25 = None
        tmp_27 = torch.functional.meshgrid(tmp_26, tmp_24, indexing='ij')
        tmp_26 = tmp_24 = None
        tmp_28 = tmp_27[0]
        tmp_29 = tmp_27[1]
        tmp_27 = None
        tmp_30 = torch.stack((tmp_29, tmp_28), -1)
        tmp_29 = tmp_28 = None
        tmp_31 = tmp_30.view(-1, 2)
        tmp_30 = None
        tmp_32 = torch.ops.aten._local_scalar_dense(tmp_11)
        tmp_11 = None
        tmp_33 = torch.full((400, 1), tmp_32, dtype=torch.float32, device=device(type='cuda', index=0))
        tmp_32 = None
        tmp_34 = torch.cat([tmp_20, tmp_31])
        tmp_20 = tmp_31 = None
        tmp_35 = torch.cat([tmp_22, tmp_33])
        tmp_22 = tmp_33 = None
        tmp_36 = tmp_34.transpose(0, 1)
        tmp_34 = None
        tmp_37 = tmp_35.transpose(0, 1)
        tmp_35 = None
        tmp_38 = tmp_9.split((64, 80), 1)
        tmp_9 = None
        tmp_39 = tmp_38[0]
        tmp_40 = tmp_38[1]
        tmp_38 = None
        tmp_41 = tmp_39.view(1, 4, 16, 2000)
        tmp_39 = None
        tmp_42 = tmp_41.transpose(2, 1)
        tmp_41 = None
        tmp_43 = tmp_42.softmax(1)
        tmp_42 = None
        tmp_44 = torch.conv2d(tmp_43, tmp_2, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_43 = tmp_2 = None
        tmp_45 = tmp_44.view(1, 4, 2000)
        tmp_44 = None
        tmp_46 = tmp_36.unsqueeze(0)
        tmp_47 = tmp_45.chunk(2, 1)
        tmp_45 = None
        tmp_48 = tmp_47[0]
        tmp_49 = tmp_47[1]
        tmp_47 = None
        tmp_50 = tmp_46 - tmp_48
        tmp_48 = None
        tmp_51 = tmp_46 + tmp_49
        tmp_46 = tmp_49 = None
        tmp_52 = tmp_50 + tmp_51
        tmp_53 = tmp_52 / 2
        tmp_52 = None
        tmp_54 = tmp_51 - tmp_50
        tmp_51 = tmp_50 = None
        tmp_55 = torch.cat((tmp_53, tmp_54), 1)
        tmp_53 = tmp_54 = None
        tmp_56 = tmp_55 * tmp_37
        tmp_55 = None
        tmp_57 = tmp_40.sigmoid()
        tmp_40 = None
        tmp_58 = torch.cat((tmp_56, tmp_57), 1)
        tmp_56 = tmp_57 = None
        return (tmp_6, tmp_36, tmp_37, tmp_58)