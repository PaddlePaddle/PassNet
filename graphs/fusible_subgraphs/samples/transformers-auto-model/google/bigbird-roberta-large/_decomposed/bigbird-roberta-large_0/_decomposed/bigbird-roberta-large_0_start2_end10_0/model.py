import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = 0.5 * in_0
        tmp_1 = torch.pow(in_0, 3.0)
        tmp_2 = 0.044715 * tmp_1
        tmp_1 = None
        tmp_3 = in_0 + tmp_2
        tmp_2 = None
        tmp_4 = 0.7978845608028654 * tmp_3
        tmp_3 = None
        tmp_5 = torch.tanh(tmp_4)
        tmp_4 = None
        tmp_6 = 1.0 + tmp_5
        tmp_5 = None
        tmp_7 = tmp_0 * tmp_6
        tmp_0 = tmp_6 = None
        return (tmp_7,)