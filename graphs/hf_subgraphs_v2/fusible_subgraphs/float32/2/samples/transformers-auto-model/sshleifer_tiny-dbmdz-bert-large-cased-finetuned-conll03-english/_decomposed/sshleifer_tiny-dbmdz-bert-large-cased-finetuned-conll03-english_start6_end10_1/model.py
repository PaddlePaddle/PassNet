import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.nn.functional.layer_norm(in_3, (2,), tmp_2, tmp_1, 1e-12)
        tmp_2 = tmp_1 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, 0.1, False, False)
        tmp_3 = None
        tmp_5 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_6 = tmp_5.expand(16, 1, 128, 128)
        tmp_5 = None
        return (tmp_4, tmp_6)