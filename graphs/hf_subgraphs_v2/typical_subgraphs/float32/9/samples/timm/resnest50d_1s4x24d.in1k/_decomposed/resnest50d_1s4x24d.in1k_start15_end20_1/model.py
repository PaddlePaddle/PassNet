import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_1 = torch.sym_sum([-1, in_0])
        tmp_2 = tmp_1 // 4
        tmp_3 = torch.sym_sum([1, tmp_2])
        tmp_2 = tmp_3 = None
        tmp_4 = tmp_0.mean((2, 3), keepdim=True)
        return (tmp_1, tmp_0, tmp_4)