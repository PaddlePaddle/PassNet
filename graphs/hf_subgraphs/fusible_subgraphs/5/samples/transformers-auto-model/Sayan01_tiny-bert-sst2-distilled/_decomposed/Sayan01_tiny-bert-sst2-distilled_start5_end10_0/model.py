import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        in_3 += in_4
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.layer_norm(tmp_3, (312,), tmp_2, tmp_1, 1e-12)
        tmp_3 = tmp_2 = tmp_1 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.1, False, False)
        tmp_4 = None
        tmp_6 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_7 = tmp_6.expand(4, 1, 512, 512)
        tmp_6 = None
        return (tmp_5, tmp_7)