import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = tmp_0.__eq__(32)
        tmp_0 = None
        tmp_2 = tmp_1.sum(-1)
        tmp_1 = None
        tmp_3 = tmp_2.float()
        tmp_2 = None
        tmp_4 = tmp_3 / in_2
        tmp_3 = None
        tmp_5 = in_1 * 0.88
        tmp_6 = 1 - tmp_4
        tmp_4 = None
        tmp_7 = tmp_6[slice(None, None, None), None, None]
        tmp_6 = None
        return (tmp_7, tmp_5)