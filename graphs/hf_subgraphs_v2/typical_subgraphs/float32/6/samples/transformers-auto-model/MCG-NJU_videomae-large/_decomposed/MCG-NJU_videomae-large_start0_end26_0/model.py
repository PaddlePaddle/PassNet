import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
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
        tmp_11 = tmp_0.permute(0, 2, 1, 3, 4)
        tmp_0 = None
        tmp_12 = torch.conv3d(tmp_11, tmp_2, tmp_1, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
        tmp_11 = tmp_2 = tmp_1 = None
        tmp_13 = tmp_12.flatten(2)
        tmp_12 = None
        tmp_14 = tmp_13.transpose(1, 2)
        tmp_13 = None
        tmp_15 = tmp_3.detach()
        tmp_3 = None
        tmp_16 = tmp_15.type_as(tmp_14)
        tmp_15 = None
        tmp_17 = tmp_16.to(device=device(type='cuda', index=0), copy=True)
        tmp_16 = None
        tmp_18 = tmp_14 + tmp_17
        tmp_14 = tmp_17 = None
        tmp_19 = torch.nn.functional.layer_norm(tmp_18, (1024,), tmp_10, tmp_9, 1e-12)
        tmp_10 = tmp_9 = None
        tmp_20 = torch.zeros_like(tmp_8, requires_grad=False)
        tmp_21 = torch.nn.functional.linear(input=tmp_19, weight=tmp_4, bias=tmp_20)
        tmp_4 = tmp_20 = None
        tmp_22 = torch.nn.functional.linear(input=tmp_19, weight=tmp_6, bias=tmp_8)
        tmp_6 = tmp_8 = None
        tmp_23 = torch.nn.functional.linear(input=tmp_19, weight=tmp_5, bias=tmp_7)
        tmp_19 = tmp_5 = tmp_7 = None
        tmp_24 = tmp_21.view(1, -1, 16, 64)
        tmp_21 = None
        tmp_25 = tmp_24.transpose(1, 2)
        tmp_24 = None
        tmp_26 = tmp_22.view(1, -1, 16, 64)
        tmp_22 = None
        tmp_27 = tmp_26.transpose(1, 2)
        tmp_26 = None
        tmp_28 = tmp_23.view(1, -1, 16, 64)
        tmp_23 = None
        tmp_29 = tmp_28.transpose(1, 2)
        tmp_28 = None
        tmp_30 = tmp_29.contiguous()
        tmp_29 = None
        tmp_31 = tmp_25.contiguous()
        tmp_25 = None
        tmp_32 = tmp_27.contiguous()
        tmp_27 = None
        tmp_33 = torch.nn.functional.scaled_dot_product_attention(tmp_30, tmp_31, tmp_32, attn_mask=None, dropout_p=0.0, scale=0.125, is_causal=False)
        tmp_30 = tmp_31 = tmp_32 = None
        tmp_34 = tmp_33.transpose(1, 2)
        tmp_33 = None
        tmp_35 = tmp_34.contiguous()
        tmp_34 = None
        tmp_36 = tmp_35.reshape((1, 1568, 1024))
        tmp_35 = None
        return (tmp_36, tmp_18)