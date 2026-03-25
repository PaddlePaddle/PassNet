import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0 + in_2
        tmp_1 = 0.7978845608028654 * tmp_0
        tmp_0 = None
        tmp_2 = torch.tanh(tmp_1)
        tmp_1 = None
        tmp_3 = 1.0 + tmp_2
        tmp_2 = None
        tmp_4 = in_1 * tmp_3
        tmp_3 = None
        return (tmp_4,)