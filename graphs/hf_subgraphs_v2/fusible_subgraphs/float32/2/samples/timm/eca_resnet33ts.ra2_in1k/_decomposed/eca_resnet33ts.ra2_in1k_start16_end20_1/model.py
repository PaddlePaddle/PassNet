import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.sym_sum([-1, in_1])
        tmp_1 = tmp_0 // 4
        tmp_2 = torch.sym_sum([1, tmp_1])
        tmp_1 = tmp_2 = None
        tmp_3 = in_0.view(1, 1, -1)
        return (tmp_0, tmp_3)