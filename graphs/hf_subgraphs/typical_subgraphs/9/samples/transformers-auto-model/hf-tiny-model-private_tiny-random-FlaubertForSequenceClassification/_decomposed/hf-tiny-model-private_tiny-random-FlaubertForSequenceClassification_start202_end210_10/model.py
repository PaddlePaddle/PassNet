import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.nn.functional.gelu(in_2)
        tmp_5 = torch.nn.functional.linear(tmp_4, tmp_1, tmp_0)
        tmp_4 = tmp_1 = tmp_0 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, p=0.1, training=False)
        tmp_5 = None
        tmp_7 = in_1 + tmp_6
        tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (32,), tmp_3, tmp_2, 1e-12)
        tmp_7 = tmp_3 = tmp_2 = None
        tmp_9 = in_0.unsqueeze(-1)
        tmp_10 = tmp_9.to(torch.float32)
        tmp_9 = None
        tmp_8 *= tmp_10
        tmp_11 = tmp_8
        tmp_8 = tmp_10 = None
        return (tmp_11,)