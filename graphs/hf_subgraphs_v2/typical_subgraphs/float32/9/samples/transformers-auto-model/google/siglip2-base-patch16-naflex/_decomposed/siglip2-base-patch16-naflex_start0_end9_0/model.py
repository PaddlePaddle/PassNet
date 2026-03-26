import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = w_6
        tmp_8 = w_7
        tmp_9 = w_8
        tmp_10 = w_9
        tmp_11 = w_10
        tmp_12 = tmp_0.view(-1, 64)
        tmp_0 = None
        tmp_13 = tmp_1[slice(None, None, None), slice(None, 64, None)]
        tmp_1 = None
        tmp_14 = torch.nn.functional.embedding(tmp_12, tmp_3, None, None, 2.0, False, False)
        tmp_12 = tmp_3 = None
        tmp_15 = torch.nn.functional.embedding(tmp_13, tmp_2, None, None, 2.0, False, False)
        tmp_13 = tmp_2 = None
        tmp_16 = tmp_14 + tmp_15
        tmp_14 = tmp_15 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (768,), tmp_5, tmp_4, 1e-06)
        tmp_5 = tmp_4 = None
        tmp_18 = torch.nn.functional.linear(tmp_17, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_19 = torch.nn.functional.linear(tmp_17, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_20 = torch.nn.functional.linear(tmp_17, tmp_11, tmp_10)
        tmp_17 = tmp_11 = tmp_10 = None
        return (tmp_16, tmp_19, tmp_18, tmp_20)