import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = in_0 * 0.5
        tmp_5 = 0.79788456 * in_0
        tmp_6 = 0.044715 * in_0
        tmp_7 = tmp_6 * in_0
        tmp_6 = None
        tmp_8 = 1 + tmp_7
        tmp_7 = None
        tmp_9 = tmp_5 * tmp_8
        tmp_5 = tmp_8 = None
        tmp_10 = torch.tanh(tmp_9)
        tmp_9 = None
        tmp_11 = 1.0 + tmp_10
        tmp_10 = None
        tmp_12 = tmp_4 * tmp_11
        tmp_4 = tmp_11 = None
        tmp_13 = torch.nn.functional.linear(tmp_12, tmp_1, tmp_0)
        tmp_12 = tmp_1 = tmp_0 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, p=0.0, training=False)
        tmp_13 = None
        tmp_15 = in_1 + tmp_14
        tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (1024,), tmp_3, tmp_2, 1e-05)
        tmp_15 = tmp_3 = tmp_2 = None
        return (tmp_16,)