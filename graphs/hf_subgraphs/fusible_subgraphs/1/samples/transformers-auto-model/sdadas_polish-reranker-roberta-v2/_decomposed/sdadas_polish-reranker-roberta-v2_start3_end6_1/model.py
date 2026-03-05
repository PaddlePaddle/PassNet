import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.to(dtype=torch.bfloat16)
        tmp_1 = 1.0 - tmp_0
        tmp_0 = None
        tmp_2 = tmp_1 * -3.3895313892515355e+38
        tmp_1 = None
        return (tmp_2,)