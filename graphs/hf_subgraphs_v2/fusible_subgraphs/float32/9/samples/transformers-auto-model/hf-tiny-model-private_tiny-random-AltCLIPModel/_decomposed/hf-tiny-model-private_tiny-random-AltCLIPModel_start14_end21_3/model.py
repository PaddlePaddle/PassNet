import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = torch.nn.functional.embedding(tmp_0, tmp_5, 1, None, 2.0, False, False)
        tmp_0 = tmp_5 = None
        tmp_7 = torch.nn.functional.embedding(in_6, tmp_4, None, None, 2.0, False, False)
        tmp_4 = None
        tmp_8 = tmp_6 + tmp_7
        tmp_6 = tmp_7 = None
        tmp_9 = torch.nn.functional.embedding(in_7, tmp_3, 1, None, 2.0, False, False)
        tmp_3 = None
        tmp_8 += tmp_9
        tmp_10 = tmp_8
        tmp_8 = tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (32,), tmp_2, tmp_1, 1e-05)
        tmp_10 = tmp_2 = tmp_1 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.1, False, False)
        tmp_11 = None
        return (tmp_12,)