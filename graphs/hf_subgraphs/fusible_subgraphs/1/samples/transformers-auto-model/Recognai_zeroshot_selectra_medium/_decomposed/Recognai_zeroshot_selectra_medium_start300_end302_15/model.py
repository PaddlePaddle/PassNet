import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_3 + in_2
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (384,), tmp_1, tmp_0, 1e-12)
        tmp_2 = tmp_1 = tmp_0 = None
        return (tmp_3,)