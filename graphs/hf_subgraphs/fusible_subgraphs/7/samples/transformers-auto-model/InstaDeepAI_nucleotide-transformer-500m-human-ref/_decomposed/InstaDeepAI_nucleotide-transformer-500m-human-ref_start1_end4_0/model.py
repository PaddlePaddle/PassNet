import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.to(dtype=torch.float32)
        tmp_1 = 1.0 - tmp_0
        tmp_0 = None
        tmp_2 = tmp_1 * -3.4028234663852886e+38
        tmp_1 = None
        return (tmp_2,)