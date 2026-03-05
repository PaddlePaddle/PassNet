import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, in_0, in_1, in_2, in_3):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = torch.nn.functional.multi_head_attention_forward(in_3, in_3, in_3, 768, 12, in_1, in_0, None, None, False, 0.0, tmp_2, tmp_1, False, None, False, in_2, use_separate_proj_weight=True, q_proj_weight=tmp_3, k_proj_weight=tmp_0, v_proj_weight=tmp_4)
        tmp_2 = tmp_1 = tmp_3 = tmp_0 = tmp_4 = None
        return (tmp_5,)