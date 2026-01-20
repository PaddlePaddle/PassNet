import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, w_0, w_1):
        in_1 += in_2
        tmp_0 = in_1
        tmp_1 = torch.nn.functional.layer_norm(tmp_0, (384,), w_1, w_0, 1e-12)
        tmp_0 = None
        tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
        tmp_1 = None
        tmp_3 = in_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_4 = tmp_3.expand(2, 1, 36, 36)
        tmp_3 = None
        return (tmp_2, tmp_4)