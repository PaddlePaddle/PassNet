import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = tmp_0.index_select(-2, in_2)
        tmp_0 = None
        tmp_2 = in_1.view((-1, 1))
        tmp_3 = tmp_2.expand_as(tmp_1)
        tmp_2 = None
        return (tmp_3, tmp_1)