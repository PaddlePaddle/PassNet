import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1):
        tmp_0 = torch.nn.functional.layer_norm(in_1, (16,), w_1, w_0, 1e-05)
        tmp_1 = in_0 + tmp_0
        tmp_0 = None
        tmp_2 = tmp_1.view(1, 16, 16, 16)
        tmp_1 = None
        tmp_3 = tmp_2[slice(None, None, None), slice(0, None, 2), slice(0, None, 2), slice(None, None, None)]
        tmp_4 = tmp_2[slice(None, None, None), slice(1, None, 2), slice(0, None, 2), slice(None, None, None)]
        tmp_5 = tmp_2[slice(None, None, None), slice(0, None, 2), slice(1, None, 2), slice(None, None, None)]
        tmp_6 = tmp_2[slice(None, None, None), slice(1, None, 2), slice(1, None, 2), slice(None, None, None)]
        tmp_2 = None
        return (tmp_3, tmp_4, tmp_5, tmp_6)