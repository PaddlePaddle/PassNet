import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = torch.nn.functional.embedding(tmp_1, tmp_6, 0, None, 2.0, False, False)
        tmp_1 = tmp_6 = None
        tmp_8 = tmp_2[slice(None, None, None), slice(None, 256, None)]
        tmp_2 = None
        tmp_9 = torch.nn.functional.embedding(tmp_8, tmp_5, None, None, 2.0, False, False)
        tmp_8 = tmp_5 = None
        tmp_10 = tmp_7 + tmp_9
        tmp_7 = tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (768,), tmp_4, tmp_3, 1e-12)
        tmp_10 = tmp_4 = tmp_3 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.1, False, False)
        tmp_11 = None
        tmp_13 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_14 = tmp_13.expand(8, 1, 256, 256)
        tmp_13 = None
        tmp_15 = tmp_14.to(torch.float32)
        tmp_14 = None
        tmp_16 = torch.tensor(1.0, dtype=torch.float32)
        tmp_17 = tmp_16 - tmp_15
        tmp_16 = tmp_15 = None
        tmp_18 = tmp_17.to(torch.bool)
        tmp_19 = tmp_17.masked_fill(tmp_18, -3.4028234663852886e+38)
        tmp_17 = tmp_18 = None
        return (tmp_19, tmp_12)