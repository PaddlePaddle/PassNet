import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = 0.5 * in_0
        tmp_5 = torch.pow(in_0, 3.0)
        tmp_6 = 0.044715 * tmp_5
        tmp_5 = None
        tmp_7 = in_0 + tmp_6
        tmp_6 = None
        tmp_8 = 0.7978845608028654 * tmp_7
        tmp_7 = None
        tmp_9 = torch.tanh(tmp_8)
        tmp_8 = None
        tmp_10 = 1.0 + tmp_9
        tmp_9 = None
        tmp_11 = tmp_4 * tmp_10
        tmp_4 = tmp_10 = None
        tmp_12 = torch.nn.functional.linear(tmp_11, tmp_1, tmp_0)
        tmp_11 = tmp_1 = tmp_0 = None
        tmp_13 = tmp_12 + in_1
        tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (1024,), tmp_3, tmp_2, 1e-12)
        tmp_13 = tmp_3 = tmp_2 = None
        return (tmp_14,)