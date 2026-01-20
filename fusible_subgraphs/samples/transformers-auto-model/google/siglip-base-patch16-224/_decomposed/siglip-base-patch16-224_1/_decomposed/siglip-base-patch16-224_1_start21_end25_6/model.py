import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_1.exp()
        tmp_1 = in_0 * tmp_0
        tmp_0 = None
        tmp_2 = tmp_1 + in_2
        tmp_1 = None
        tmp_3 = tmp_2.t()
        return (tmp_2, tmp_3)