import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.layer_norm(in_2, (32,), tmp_1, tmp_0, 1e-06)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.view(1, 64, 48, 32)
        tmp_2 = None
        return (tmp_3,)