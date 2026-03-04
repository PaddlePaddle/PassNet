import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3 + in_5
        tmp_4 = torch.nn.functional.embedding(in_4, tmp_2, 1, None, 2.0, False, False)
        tmp_2 = None
        tmp_3 += tmp_4
        tmp_5 = tmp_3
        tmp_3 = tmp_4 = None
        tmp_6 = torch.nn.functional.layer_norm(tmp_5, (1024,), tmp_1, tmp_0, 1e-05)
        tmp_5 = tmp_1 = tmp_0 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.1, False, False)
        tmp_6 = None
        return (tmp_7,)