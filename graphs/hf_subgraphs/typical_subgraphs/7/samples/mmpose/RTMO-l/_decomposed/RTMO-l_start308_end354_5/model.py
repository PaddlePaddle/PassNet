import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13):
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
        tmp_13 = in_13.flatten(2)
        tmp_14 = tmp_13.permute(0, 2, 1)
        tmp_13 = None
        tmp_15 = tmp_14.contiguous()
        tmp_14 = None
        tmp_16 = torch.arange(8, dtype=torch.float32, device=device(type='cuda', index=0))
        tmp_17 = torch.arange(8, dtype=torch.float32, device=device(type='cuda', index=0))
        tmp_18 = torch.functional.meshgrid(tmp_16, tmp_17)
        tmp_16 = tmp_17 = None
        tmp_19 = tmp_18[0]
        tmp_20 = tmp_18[1]
        tmp_18 = None
        tmp_21 = tmp_19.flatten()
        tmp_19 = None
        tmp_22 = tmp_20.flatten()
        tmp_20 = None
        tmp_23 = tmp_12.reshape(1, -1)
        tmp_12 = None
        tmp_24 = tmp_21.unsqueeze(-1)
        tmp_21 = None
        tmp_25 = tmp_24 / tmp_23
        tmp_24 = None
        tmp_26 = tmp_22.unsqueeze(-1)
        tmp_22 = None
        tmp_27 = tmp_26 / tmp_23
        tmp_26 = tmp_23 = None
        tmp_28 = tmp_25.cos()
        tmp_29 = tmp_25.sin()
        tmp_25 = None
        tmp_30 = torch.cat((tmp_28, tmp_29), dim=-1)
        tmp_28 = tmp_29 = None
        tmp_31 = tmp_27.cos()
        tmp_32 = tmp_27.sin()
        tmp_27 = None
        tmp_33 = torch.cat((tmp_31, tmp_32), dim=-1)
        tmp_31 = tmp_32 = None
        tmp_34 = torch.stack((tmp_30, tmp_33), dim=-1)
        tmp_30 = tmp_33 = None
        tmp_35 = tmp_34.transpose(-1, -2)
        tmp_34 = None
        tmp_36 = tmp_35.reshape(1, 64, -1)
        tmp_35 = None
        tmp_37 = tmp_15 + tmp_36
        tmp_38 = tmp_15 + tmp_36
        tmp_36 = None
        tmp_39 = tmp_37.transpose(0, 1)
        tmp_37 = None
        tmp_40 = tmp_38.transpose(0, 1)
        tmp_38 = None
        tmp_41 = tmp_15.transpose(0, 1)
        tmp_42 = torch.nn.functional.multi_head_attention_forward(tmp_39, tmp_40, tmp_41, 256, 8, tmp_11, tmp_10, None, None, False, 0.0, tmp_9, tmp_8, training=False, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False)
        tmp_39 = tmp_40 = tmp_41 = tmp_11 = tmp_10 = tmp_9 = tmp_8 = None
        tmp_43 = tmp_42[0]
        tmp_42 = None
        tmp_44 = tmp_43.transpose(0, 1)
        tmp_43 = None
        tmp_45 = torch.nn.functional.dropout(tmp_44, 0.0, False, False)
        tmp_44 = None
        tmp_46 = torch.nn.functional.dropout(tmp_45, 0.0, False, False)
        tmp_45 = None
        tmp_47 = tmp_15 + tmp_46
        tmp_15 = tmp_46 = None
        tmp_48 = torch.nn.functional.layer_norm(tmp_47, (256,), tmp_5, tmp_4, 1e-05)
        tmp_47 = tmp_5 = tmp_4 = None
        tmp_49 = torch.nn.functional.linear(tmp_48, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_50 = torch.nn.functional.gelu(tmp_49, approximate='none')
        tmp_49 = None
        tmp_51 = torch.nn.functional.dropout(tmp_50, 0.0, False, False)
        tmp_50 = None
        tmp_52 = torch.nn.functional.linear(tmp_51, tmp_3, tmp_2)
        tmp_51 = tmp_3 = tmp_2 = None
        tmp_53 = torch.nn.functional.dropout(tmp_52, 0.0, False, False)
        tmp_52 = None
        tmp_54 = tmp_48 + tmp_53
        tmp_48 = tmp_53 = None
        tmp_55 = torch.nn.functional.layer_norm(tmp_54, (256,), tmp_7, tmp_6, 1e-05)
        tmp_54 = tmp_7 = tmp_6 = None
        tmp_56 = tmp_55.permute(0, 2, 1)
        tmp_55 = None
        tmp_57 = tmp_56.contiguous()
        tmp_56 = None
        tmp_58 = tmp_57.view([-1, 256, 8, 8])
        tmp_57 = None
        return (tmp_58,)