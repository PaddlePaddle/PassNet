import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.to(torch.float32)
        tmp_1 = 1.0 - tmp_0
        tmp_0 = None
        tmp_2 = tmp_1.bool()
        tmp_3 = tmp_1.masked_fill(tmp_2, -3.4028234663852886e+38)
        tmp_2 = None
        tmp_4 = tmp_3 * tmp_1
        tmp_3 = tmp_1 = None
        return (tmp_4,)