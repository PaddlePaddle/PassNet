import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = tmp_0 * in_2
        tmp_0 = None
        tmp_3 = tmp_2 + in_3
        tmp_2 = None
        tmp_4 = tmp_1[slice(None, 729, None)]
        tmp_1 = None
        tmp_5 = tmp_4.reshape(1, 27, 27, -1)
        tmp_4 = None
        tmp_6 = tmp_5.permute(0, 3, 1, 2)
        tmp_5 = None
        return (tmp_3, tmp_6)