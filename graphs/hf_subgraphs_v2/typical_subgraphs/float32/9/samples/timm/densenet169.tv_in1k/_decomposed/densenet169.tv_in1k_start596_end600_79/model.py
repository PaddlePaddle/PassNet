import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.adaptive_avg_pool2d(in_0, 1)
        tmp_3 = tmp_2.flatten(1, -1)
        tmp_2 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
        tmp_3 = None
        tmp_5 = torch.nn.functional.linear(tmp_4, tmp_1, tmp_0)
        tmp_4 = tmp_1 = tmp_0 = None
        return (tmp_5,)