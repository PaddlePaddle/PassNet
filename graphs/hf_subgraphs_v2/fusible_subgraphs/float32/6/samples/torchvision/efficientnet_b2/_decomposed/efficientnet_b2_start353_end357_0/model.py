import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
        tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
        tmp_0 = None
        tmp_2 = torch.flatten(tmp_1, 1)
        tmp_1 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.3, False, True)
        tmp_2 = None
        return (tmp_3,)