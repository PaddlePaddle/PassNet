import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor):
        tmp_11 = in_0.permute(0, 2, 1, 3, 4);  in_0 = None
        conv3d = torch.conv3d(tmp_11, in_2, in_1, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1);  tmp_11 = in_2 = in_1 = None
        tmp_13 = conv3d.flatten(2);  conv3d = None
        tmp_14 = tmp_13.transpose(1, 2);  tmp_13 = None
        tmp_15 = in_3.detach();  in_3 = None
        tmp_16 = tmp_15.type_as(tmp_14);  tmp_15 = None
        tmp_17 = tmp_16.to(device = device(type='cuda', index=0), copy = True);  tmp_16 = None
        tmp_18 = tmp_14 + tmp_17;  tmp_14 = tmp_17 = None
        tmp_19 = torch.nn.functional.layer_norm(tmp_18, (768,), in_10, in_9, 1e-12);  in_10 = in_9 = None
        tmp_20 = torch.zeros_like(in_8, requires_grad = False)
        linear = torch.nn.functional.linear(input = tmp_19, weight = in_4, bias = tmp_20);  in_4 = tmp_20 = None
        linear_1 = torch.nn.functional.linear(input = tmp_19, weight = in_6, bias = in_8);  in_6 = in_8 = None
        linear_2 = torch.nn.functional.linear(input = tmp_19, weight = in_5, bias = in_7);  tmp_19 = in_5 = in_7 = None
        tmp_24 = linear.view(1, -1, 12, 64);  linear = None
        tmp_25 = tmp_24.transpose(1, 2);  tmp_24 = None
        tmp_26 = linear_1.view(1, -1, 12, 64);  linear_1 = None
        tmp_27 = tmp_26.transpose(1, 2);  tmp_26 = None
        tmp_28 = linear_2.view(1, -1, 12, 64);  linear_2 = None
        tmp_29 = tmp_28.transpose(1, 2);  tmp_28 = None
        tmp_30 = tmp_29.contiguous();  tmp_29 = None
        tmp_31 = tmp_25.contiguous();  tmp_25 = None
        tmp_32 = tmp_27.contiguous();  tmp_27 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_30, tmp_31, tmp_32, attn_mask = None, dropout_p = 0.0, scale = 0.125, is_causal = False);  tmp_30 = tmp_31 = tmp_32 = None
        tmp_34 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_35 = tmp_34.contiguous();  tmp_34 = None
        tmp_36 = tmp_35.reshape((1, 1568, 768));  tmp_35 = None
        return (tmp_36, tmp_18)
        