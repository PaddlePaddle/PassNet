import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.nn.functional.multi_head_attention_forward(in_1, in_0, in_0, 768, 24, tmp_3, tmp_2, None, None, False, 0.0, tmp_1, tmp_0, training=False, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False)
        tmp_3 = tmp_2 = tmp_1 = tmp_0 = None
        return (tmp_4,)