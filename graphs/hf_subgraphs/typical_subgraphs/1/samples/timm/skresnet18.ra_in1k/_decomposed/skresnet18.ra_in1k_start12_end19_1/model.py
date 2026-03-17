import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.nn.functional.relu(in_2, inplace=True)
        tmp_1 = torch.stack([in_1, tmp_0], dim=1)
        tmp_0 = None
        tmp_2 = torch.sym_sum([-1, in_0])
        tmp_3 = tmp_2 // 4
        tmp_4 = torch.sym_sum([1, tmp_3])
        tmp_3 = tmp_4 = None
        tmp_5 = tmp_1.sum(1)
        tmp_6 = tmp_5.mean((2, 3), keepdim=True)
        tmp_5 = None
        return (tmp_1, tmp_2, tmp_6)