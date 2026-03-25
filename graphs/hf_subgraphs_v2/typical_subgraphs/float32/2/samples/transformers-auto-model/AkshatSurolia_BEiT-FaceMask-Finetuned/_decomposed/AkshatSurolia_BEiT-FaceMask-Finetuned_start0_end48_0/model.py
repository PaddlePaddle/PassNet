import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = torch.conv2d(tmp_0, tmp_2, tmp_1, (16, 16), (0, 0), (1, 1), 1)
        tmp_0 = tmp_2 = tmp_1 = None
        tmp_8 = tmp_7.flatten(2)
        tmp_7 = None
        tmp_9 = tmp_8.transpose(1, 2)
        tmp_8 = None
        tmp_10 = tmp_3.expand(1, -1, -1)
        tmp_3 = None
        tmp_11 = torch.cat((tmp_10, tmp_9), dim=1)
        tmp_10 = tmp_9 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
        tmp_11 = None
        tmp_13 = tmp_6[slice(None, 729, None)]
        tmp_14 = tmp_13.reshape(1, 27, 27, -1)
        tmp_13 = None
        tmp_15 = tmp_14.permute(0, 3, 1, 2)
        tmp_14 = None
        tmp_16 = torch.nn.functional.interpolate(tmp_15, size=(27, 27), mode='bilinear')
        tmp_15 = None
        tmp_17 = tmp_16.permute(0, 2, 3, 1)
        tmp_16 = None
        tmp_18 = tmp_17.reshape(729, -1)
        tmp_17 = None
        tmp_19 = tmp_6[slice(729, None, None)]
        tmp_6 = None
        tmp_20 = torch.cat([tmp_18, tmp_19])
        tmp_18 = tmp_19 = None
        tmp_21 = torch.arange(14)
        tmp_22 = torch.arange(14)
        tmp_23 = torch.functional.meshgrid(tmp_21, tmp_22, indexing='ij')
        tmp_21 = tmp_22 = None
        tmp_24 = tmp_23[0]
        tmp_25 = tmp_23[1]
        tmp_23 = None
        tmp_26 = torch.stack((tmp_24, tmp_25))
        tmp_24 = tmp_25 = None
        tmp_27 = torch.flatten(tmp_26, 1)
        tmp_26 = None
        tmp_28 = tmp_27[slice(None, None, None), slice(None, None, None), None]
        tmp_29 = tmp_27[slice(None, None, None), None, slice(None, None, None)]
        tmp_27 = None
        tmp_30 = tmp_28 - tmp_29
        tmp_28 = tmp_29 = None
        tmp_31 = tmp_30.permute(1, 2, 0)
        tmp_30 = None
        tmp_32 = tmp_31.contiguous()
        tmp_31 = None
        tmp_33 = tmp_32[slice(None, None, None), slice(None, None, None), 0]
        tmp_33 += 13
        tmp_34 = tmp_33
        tmp_33 = None
        tmp_32[slice(None, None, None), slice(None, None, None), 0] = tmp_34
        tmp_35 = tmp_32
        tmp_34 = tmp_35 = None
        tmp_36 = tmp_32[slice(None, None, None), slice(None, None, None), 1]
        tmp_36 += 13
        tmp_37 = tmp_36
        tmp_36 = None
        tmp_32[slice(None, None, None), slice(None, None, None), 1] = tmp_37
        tmp_38 = tmp_32
        tmp_37 = tmp_38 = None
        tmp_39 = tmp_32[slice(None, None, None), slice(None, None, None), 0]
        tmp_39 *= 27
        tmp_40 = tmp_39
        tmp_39 = None
        tmp_32[slice(None, None, None), slice(None, None, None), 0] = tmp_40
        tmp_41 = tmp_32
        tmp_40 = tmp_41 = None
        tmp_42 = torch.zeros(size=(197, 197), dtype=torch.int64)
        tmp_43 = tmp_32.sum(-1)
        tmp_32 = None
        tmp_42[slice(1, None, None), slice(1, None, None)] = tmp_43
        tmp_44 = tmp_42
        tmp_43 = tmp_44 = None
        tmp_42[0, slice(0, None, None)] = 729
        tmp_45 = tmp_42
        tmp_45 = None
        tmp_42[slice(0, None, None), 0] = 730
        tmp_46 = tmp_42
        tmp_46 = None
        tmp_42[0, 0] = 731
        tmp_47 = tmp_42
        tmp_47 = None
        tmp_48 = tmp_42.view(-1)
        tmp_42 = None
        tmp_49 = tmp_20[tmp_48]
        tmp_20 = tmp_48 = None
        tmp_50 = tmp_49.view(197, 197, -1)
        tmp_49 = None
        tmp_51 = tmp_50.permute(2, 0, 1)
        tmp_50 = None
        tmp_52 = tmp_51.contiguous()
        tmp_51 = None
        tmp_53 = tmp_52.unsqueeze(0)
        tmp_52 = None
        tmp_54 = torch.nn.functional.layer_norm(tmp_12, (768,), tmp_5, tmp_4, 1e-12)
        tmp_5 = tmp_4 = None
        return (tmp_12, tmp_54, tmp_53)