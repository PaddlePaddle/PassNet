import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0
        tmp_1 = tmp_0.ne(1)
        tmp_0 = None
        tmp_2 = tmp_1.int()
        tmp_1 = None
        tmp_3 = torch.cumsum(tmp_2, dim=1)
        tmp_4 = tmp_3.type_as(tmp_2)
        tmp_3 = None
        tmp_5 = tmp_4 * tmp_2
        tmp_4 = tmp_2 = None
        tmp_6 = tmp_5.long()
        tmp_5 = None
        tmp_7 = tmp_6 + 1
        tmp_6 = None
        return (tmp_7,)