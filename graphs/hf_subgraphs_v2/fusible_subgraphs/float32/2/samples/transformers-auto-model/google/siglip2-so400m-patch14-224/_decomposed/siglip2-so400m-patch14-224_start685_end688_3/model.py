import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.multi_head_attention_forward(in_5, in_4, in_4, 1152, 16, tmp_3, tmp_2, None, None, False, 0.0, tmp_1, tmp_0, training=False, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False)
        tmp_3 = tmp_2 = tmp_1 = tmp_0 = None
        tmp_5 = tmp_4[0]
        tmp_4 = None
        tmp_6 = tmp_5.transpose(1, 0)
        tmp_5 = None
        return (tmp_6,)