import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.to(torch.bfloat16)
        tmp_1 = torch.tensor(1.0, dtype=torch.bfloat16)
        tmp_2 = tmp_1 - tmp_0
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.to(torch.bool)
        tmp_4 = tmp_2.masked_fill(tmp_3, -3.3895313892515355e+38)
        tmp_2 = tmp_3 = None
        return (tmp_4,)