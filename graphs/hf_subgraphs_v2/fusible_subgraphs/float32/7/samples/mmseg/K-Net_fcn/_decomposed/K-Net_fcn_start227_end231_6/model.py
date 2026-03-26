import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.multi_head_attention_forward(in_4, in_4, in_4, 512, 8, tmp_3, tmp_2, None, None, False, 0.0, tmp_1, tmp_0, training=False, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False)
        tmp_3 = tmp_2 = tmp_1 = tmp_0 = None
        tmp_5 = tmp_4[0]
        tmp_4 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
        tmp_5 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
        tmp_6 = None
        return (tmp_7,)