import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.nn.functional.embedding(tmp_1, tmp_2, 1, None, 2.0, False, False)
        tmp_2 = None
        tmp_4 = tmp_1.__eq__(2)
        tmp_5 = tmp_4.unsqueeze(-1)
        tmp_4 = None
        tmp_6 = tmp_3.masked_fill(tmp_5, 0.0)
        tmp_3 = tmp_5 = None
        tmp_7 = tmp_0.sum(-1)
        tmp_0 = None
        tmp_8 = tmp_1.__eq__(2)
        tmp_1 = None
        tmp_9 = tmp_8.sum(-1)
        tmp_8 = None
        tmp_10 = tmp_9.float()
        tmp_9 = None
        tmp_11 = tmp_10 / tmp_7
        tmp_10 = tmp_7 = None
        tmp_12 = tmp_6 * 0.88
        tmp_6 = None
        tmp_13 = 1 - tmp_11
        tmp_11 = None
        tmp_14 = tmp_13[slice(None, None, None), None, None]
        tmp_13 = None
        return (tmp_14, tmp_12)