import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1.contiguous()
        tmp_2 = tmp_1.reshape(8, -1, 64, 64)
        tmp_1 = None
        tmp_3 = tmp_2 * tmp_0
        tmp_2 = tmp_0 = None
        tmp_4 = tmp_3 + in_2
        tmp_3 = None
        return (tmp_4,)