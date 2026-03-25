import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = w_0
        tmp_3 = w_1
        tmp_4 = w_2
        tmp_5 = w_3
        tmp_6 = in_2
        tmp_7 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_8 = tmp_7.to(dtype=torch.float32)
        tmp_7 = None
        tmp_9 = 1.0 - tmp_8
        tmp_8 = None
        tmp_10 = tmp_9 * -3.4028234663852886e+38
        tmp_9 = None
        tmp_11 = torch.nn.functional.embedding(tmp_1, tmp_3, 0, None, 2.0, False, False)
        tmp_1 = tmp_3 = None
        tmp_12 = torch.nn.functional.embedding(tmp_6, tmp_2, None, None, 2.0, False, False)
        tmp_6 = tmp_2 = None
        tmp_13 = tmp_11 + tmp_12
        tmp_11 = tmp_12 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.1, False, False)
        tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (384,), tmp_5, tmp_4, 1e-12)
        tmp_5 = tmp_4 = None
        return (tmp_14, tmp_10, tmp_15)