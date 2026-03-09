import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch._native_multi_head_attention(in_0, in_0, in_0, 1280, 16, tmp_3, tmp_2, tmp_1, tmp_0, None, False, True, None)
        tmp_3 = tmp_2 = tmp_1 = tmp_0 = None
        return (tmp_4,)