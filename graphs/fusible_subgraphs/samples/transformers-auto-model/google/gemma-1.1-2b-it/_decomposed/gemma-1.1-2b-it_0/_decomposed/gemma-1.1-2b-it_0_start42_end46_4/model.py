import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0):
        tmp_0 = w_0.float()
        tmp_1 = 1.0 + tmp_0
        tmp_0 = None
        tmp_2 = in_1 * tmp_1
        tmp_1 = None
        tmp_3 = tmp_2.type_as(in_0)
        tmp_2 = None
        return (tmp_3,)