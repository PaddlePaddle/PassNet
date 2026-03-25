import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.group_norm(in_2, 512, tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.nn.functional.gelu(tmp_2)
        tmp_2 = None
        return (tmp_3,)