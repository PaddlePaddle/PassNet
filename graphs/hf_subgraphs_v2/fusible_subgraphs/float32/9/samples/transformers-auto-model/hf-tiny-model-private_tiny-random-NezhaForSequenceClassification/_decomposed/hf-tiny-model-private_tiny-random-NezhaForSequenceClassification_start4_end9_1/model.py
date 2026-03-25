import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = torch.nn.functional.embedding(tmp_0, tmp_4, 0, None, 2.0, False, False)
        tmp_0 = tmp_4 = None
        tmp_7 = torch.nn.functional.embedding(tmp_5, tmp_3, None, None, 2.0, False, False)
        tmp_5 = tmp_3 = None
        tmp_8 = tmp_6 + tmp_7
        tmp_6 = tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (32,), tmp_2, tmp_1, 1e-12)
        tmp_8 = tmp_2 = tmp_1 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.1, False, False)
        tmp_9 = None
        return (tmp_10,)