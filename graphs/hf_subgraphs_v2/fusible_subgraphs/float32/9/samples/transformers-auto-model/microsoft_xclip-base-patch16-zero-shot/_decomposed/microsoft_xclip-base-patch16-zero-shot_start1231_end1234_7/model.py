import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_3 + in_1
        tmp_2 = tmp_0 * tmp_1
        tmp_0 = tmp_1 = None
        tmp_3 = in_2 + tmp_2
        tmp_2 = None
        return (tmp_3,)