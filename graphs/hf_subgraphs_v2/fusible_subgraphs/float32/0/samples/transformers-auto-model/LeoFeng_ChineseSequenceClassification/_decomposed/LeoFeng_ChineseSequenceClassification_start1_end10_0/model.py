import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = torch.nn.functional.embedding(tmp_1, tmp_6, 1, None, 2.0, False, False)
        tmp_1 = tmp_6 = None
        tmp_9 = torch.nn.functional.embedding(tmp_7, tmp_5, None, None, 2.0, False, False)
        tmp_7 = tmp_5 = None
        tmp_10 = tmp_8 + tmp_9
        tmp_8 = tmp_9 = None
        tmp_11 = torch.nn.functional.embedding(in_8, tmp_4, None, None, 2.0, False, False)
        tmp_4 = None
        tmp_10 += tmp_11
        tmp_12 = tmp_10
        tmp_10 = tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), tmp_3, tmp_2, 1e-12)
        tmp_12 = tmp_3 = tmp_2 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.1, False, False)
        tmp_13 = None
        tmp_15 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_16 = tmp_15.expand(1, 1, 64, 64)
        tmp_15 = None
        return (tmp_14, tmp_16)