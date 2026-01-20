import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2):
        tmp_0 = in_1 * w_2
        tmp_1 = tmp_0 + in_0
        tmp_0 = None
        tmp_2 = torch.nn.functional.layer_norm(tmp_1, (384,), w_1, w_0, 1e-06)
        tmp_1 = None
        tmp_3 = tmp_2[slice(None, None, None), 0, slice(None, None, None)]
        return (tmp_2, tmp_3)