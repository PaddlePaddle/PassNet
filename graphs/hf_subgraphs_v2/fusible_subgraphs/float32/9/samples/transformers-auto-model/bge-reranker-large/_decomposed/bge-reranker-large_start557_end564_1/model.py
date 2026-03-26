import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0.to(torch.float32)
        tmp_1 = in_1 * tmp_0
        tmp_2 = torch.sum(tmp_1, 1)
        tmp_1 = None
        tmp_3 = tmp_0.sum(1)
        tmp_0 = None
        tmp_4 = torch.clamp(tmp_3, min=1e-09)
        tmp_3 = None
        tmp_5 = tmp_2 / tmp_4
        tmp_2 = tmp_4 = None
        tmp_6 = torch.cat([tmp_5], 1)
        tmp_5 = None
        return (tmp_6,)