import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4 + in_6
        tmp_5 = torch.nn.functional.embedding(in_5, tmp_3, 1, None, 2.0, False, False)
        tmp_3 = None
        tmp_4 += tmp_5
        tmp_6 = tmp_4
        tmp_4 = tmp_5 = None
        tmp_7 = torch.nn.functional.layer_norm(tmp_6, (768,), tmp_2, tmp_1, 1e-05)
        tmp_6 = tmp_2 = tmp_1 = None
        tmp_8 = torch.nn.functional.dropout(tmp_7, 0.1, False, False)
        tmp_7 = None
        tmp_9 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_10 = tmp_9.expand(16, 1, 128, 128)
        tmp_9 = None
        return (tmp_8, tmp_10)