import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = in_1 * 0.458
        tmp_2 = tmp_1 + -0.030000000000000027
        tmp_1 = None
        tmp_3 = tmp_0[slice(None, None, None), 1]
        tmp_4 = torch.unsqueeze(tmp_3, 1)
        tmp_3 = None
        tmp_5 = tmp_4 * 0.448
        tmp_4 = None
        tmp_6 = tmp_5 + -0.08799999999999997
        tmp_5 = None
        tmp_7 = tmp_0[slice(None, None, None), 2]
        tmp_0 = None
        tmp_8 = torch.unsqueeze(tmp_7, 1)
        tmp_7 = None
        tmp_9 = tmp_8 * 0.45
        tmp_8 = None
        tmp_10 = tmp_9 + -0.18799999999999994
        tmp_9 = None
        tmp_11 = torch.cat((tmp_2, tmp_6, tmp_10), 1)
        tmp_2 = tmp_6 = tmp_10 = None
        return (tmp_11,)