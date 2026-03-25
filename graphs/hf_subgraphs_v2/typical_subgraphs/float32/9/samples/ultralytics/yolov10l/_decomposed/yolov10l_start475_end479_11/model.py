import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
        tmp_1 = in_1.detach()
        tmp_2 = in_2.detach()
        tmp_3 = tmp_0.detach()
        return (tmp_1, tmp_2, tmp_3, tmp_0)