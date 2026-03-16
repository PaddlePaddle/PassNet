import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.linear(in_5, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_5 = tmp_4.view((4, 512, 16, 64))
        tmp_4 = None
        tmp_6 = tmp_5.permute(0, 2, 1, 3)
        tmp_5 = None
        tmp_7 = torch.nn.functional.linear(in_5, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_8 = tmp_7.view((4, 512, 16, 64))
        tmp_7 = None
        tmp_9 = tmp_8.permute(0, 2, 1, 3)
        tmp_8 = None
        tmp_10 = in_6.view((4, 512, 16, 64))
        tmp_11 = tmp_10.permute(0, 2, 1, 3)
        tmp_10 = None
        tmp_12 = tmp_11.contiguous()
        tmp_11 = None
        tmp_13 = tmp_6.contiguous()
        tmp_6 = None
        tmp_14 = tmp_9.contiguous()
        tmp_9 = None
        tmp_15 = torch.nn.functional.scaled_dot_product_attention(tmp_12, tmp_13, tmp_14, attn_mask=in_4, dropout_p=0.0, is_causal=False)
        tmp_12 = tmp_13 = tmp_14 = None
        tmp_16 = tmp_15.transpose(1, 2)
        tmp_15 = None
        tmp_17 = tmp_16.reshape(4, 512, 1024)
        tmp_16 = None
        return (tmp_17,)