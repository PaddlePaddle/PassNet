import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3[slice(None, None, None), 0]
        tmp_4 = torch.nn.functional.linear(tmp_3, tmp_2, tmp_1)
        tmp_3 = tmp_2 = tmp_1 = None
        tmp_5 = torch.tanh(tmp_4)
        tmp_4 = tmp_5 = None
        tmp_6 = tmp_0.unsqueeze(-1)
        tmp_0 = None
        tmp_7 = tmp_6.expand((1, 10, 1024))
        tmp_6 = None
        return (tmp_7,)