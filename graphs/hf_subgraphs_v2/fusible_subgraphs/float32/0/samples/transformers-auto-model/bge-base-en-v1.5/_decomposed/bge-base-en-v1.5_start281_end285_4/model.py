import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2[slice(None, None, None), 0]
        tmp_3 = torch.nn.functional.linear(tmp_2, tmp_1, tmp_0)
        tmp_2 = tmp_1 = tmp_0 = None
        tmp_4 = torch.tanh(tmp_3)
        tmp_3 = tmp_4 = None
        tmp_5 = in_2[slice(None, None, None), 0]
        return (tmp_5,)