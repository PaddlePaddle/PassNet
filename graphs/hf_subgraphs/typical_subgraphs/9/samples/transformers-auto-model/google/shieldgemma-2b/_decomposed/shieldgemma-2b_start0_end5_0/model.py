import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0):
        tmp_0 = w_0
        tmp_1 = in_0
        tmp_2 = tmp_1[slice(None, None, None), slice(0, None, None), slice(None, None, None)]
        tmp_1 = None
        tmp_3 = torch.nn.functional.linear(tmp_2, tmp_0, None)
        tmp_2 = tmp_0 = None
        tmp_4 = tmp_3 / 30.0
        tmp_3 = None
        tmp_5 = torch.tanh(tmp_4)
        tmp_4 = None
        tmp_6 = tmp_5 * 30.0
        tmp_5 = None
        return (tmp_6,)