import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.silu(in_2, inplace=True)
        tmp_3 = torch.nn.functional.adaptive_avg_pool2d(tmp_2, 1)
        tmp_2 = None
        tmp_4 = torch.flatten(tmp_3, 1)
        tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.5, False, True)
        tmp_4 = None
        tmp_6 = torch.nn.functional.linear(tmp_5, tmp_1, tmp_0)
        tmp_5 = tmp_1 = tmp_0 = None
        return (tmp_6,)