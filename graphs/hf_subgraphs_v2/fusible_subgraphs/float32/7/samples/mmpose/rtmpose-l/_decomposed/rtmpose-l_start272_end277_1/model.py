import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.functional.norm(in_1, dim=-1, keepdim=True)
        tmp_2 = tmp_1 * 0.0625
        tmp_1 = None
        tmp_3 = tmp_2.clamp(min=1e-05)
        tmp_2 = None
        tmp_4 = in_1 / tmp_3
        tmp_3 = None
        tmp_5 = tmp_4 * tmp_0
        tmp_4 = tmp_0 = None
        return (tmp_5,)