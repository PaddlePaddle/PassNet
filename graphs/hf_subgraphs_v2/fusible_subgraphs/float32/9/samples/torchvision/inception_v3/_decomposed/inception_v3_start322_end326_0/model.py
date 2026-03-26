import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
        tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
        tmp_0 = None
        tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
        tmp_1 = None
        tmp_3 = torch.flatten(tmp_2, 1)
        tmp_2 = None
        return (tmp_3,)