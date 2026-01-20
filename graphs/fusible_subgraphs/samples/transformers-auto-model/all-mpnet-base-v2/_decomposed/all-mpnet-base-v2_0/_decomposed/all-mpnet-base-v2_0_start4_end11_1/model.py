import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.ne(1)
        tmp_1 = tmp_0.int()
        tmp_0 = None
        tmp_2 = torch.cumsum(tmp_1, dim=1)
        tmp_3 = tmp_2.type_as(tmp_1)
        tmp_2 = None
        tmp_4 = tmp_3 * tmp_1
        tmp_3 = tmp_1 = None
        tmp_5 = tmp_4.long()
        tmp_4 = None
        tmp_6 = tmp_5 + 1
        tmp_5 = None
        return (tmp_6,)