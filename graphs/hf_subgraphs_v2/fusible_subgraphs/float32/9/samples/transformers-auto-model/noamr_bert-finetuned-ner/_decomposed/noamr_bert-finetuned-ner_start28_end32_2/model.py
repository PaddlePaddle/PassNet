import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.embedding(in_4, tmp_1, None, None, 2.0, False, False)
        tmp_1 = None
        tmp_5 = tmp_0 + tmp_4
        tmp_0 = tmp_4 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, p=0.1, training=False)
        tmp_5 = None
        tmp_7 = torch.nn.functional.layer_norm(tmp_6, (1024,), tmp_3, tmp_2, 1e-05)
        tmp_3 = tmp_2 = None
        return (tmp_6, tmp_7)