import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0 * 0.5
        tmp_1 = 0.79788456 * in_0
        tmp_2 = 0.044715 * in_0
        tmp_3 = tmp_2 * in_0
        tmp_2 = None
        tmp_4 = 1 + tmp_3
        tmp_3 = None
        tmp_5 = tmp_1 * tmp_4
        tmp_1 = tmp_4 = None
        tmp_6 = torch.tanh(tmp_5)
        tmp_5 = None
        tmp_7 = 1.0 + tmp_6
        tmp_6 = None
        tmp_8 = tmp_0 * tmp_7
        tmp_0 = tmp_7 = None
        return (tmp_8,)