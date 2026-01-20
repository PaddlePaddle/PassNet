import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0.__eq__(32)
        tmp_1 = tmp_0.sum(-1)
        tmp_0 = None
        tmp_2 = tmp_1.float()
        tmp_1 = None
        tmp_3 = tmp_2 / in_2
        tmp_2 = None
        tmp_4 = in_1 * 0.88
        tmp_5 = 1 - tmp_3
        tmp_3 = None
        tmp_6 = tmp_5[slice(None, None, None), None, None]
        tmp_5 = None
        return (tmp_4, tmp_6)