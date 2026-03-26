import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2 + in_3
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (192,), tmp_1, tmp_0, 1e-06)
        tmp_2 = tmp_1 = tmp_0 = None
        tmp_4 = tmp_3[slice(None, None, None), 0]
        tmp_5 = tmp_3[slice(None, None, None), 1]
        tmp_3 = None
        return (tmp_4, tmp_5)