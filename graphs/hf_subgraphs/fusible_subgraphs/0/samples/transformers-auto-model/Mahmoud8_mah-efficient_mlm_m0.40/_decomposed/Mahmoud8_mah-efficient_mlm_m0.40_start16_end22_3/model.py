import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5 + in_7
        tmp_6 = torch.nn.functional.embedding(in_6, tmp_2, 1, None, 2.0, False, False)
        tmp_2 = None
        tmp_5 += tmp_6
        tmp_7 = tmp_5
        tmp_5 = tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (1024,), tmp_1, tmp_0, 1e-05)
        tmp_7 = tmp_1 = tmp_0 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False)
        tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (1024,), tmp_4, tmp_3, 1e-05)
        tmp_4 = tmp_3 = None
        return (tmp_9, tmp_10)