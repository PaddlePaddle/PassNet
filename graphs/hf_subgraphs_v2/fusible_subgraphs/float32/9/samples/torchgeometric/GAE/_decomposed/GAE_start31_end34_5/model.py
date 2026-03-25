import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0[1]
        tmp_1 = in_0[0]
        tmp_2 = in_1.index_select(-2, tmp_1)
        tmp_1 = None
        return (tmp_0, tmp_2)