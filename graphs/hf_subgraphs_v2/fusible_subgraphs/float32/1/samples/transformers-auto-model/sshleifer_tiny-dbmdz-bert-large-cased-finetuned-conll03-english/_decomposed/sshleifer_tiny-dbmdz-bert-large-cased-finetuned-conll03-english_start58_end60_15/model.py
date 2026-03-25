import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.layer_norm(in_2, (2,), tmp_1, tmp_0, 1e-12)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2[slice(None, None, None), 0]
        return (tmp_3, tmp_2)