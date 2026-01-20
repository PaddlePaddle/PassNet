import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, w_0, w_1, w_2):
        tmp_0 = in_1 + in_2
        tmp_1 = torch.nn.functional.embedding(in_0, w_2, 1, None, 2.0, False, False)
        tmp_0 += tmp_1
        tmp_2 = tmp_0
        tmp_0 = tmp_1 = None
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (32,), w_1, w_0, 1e-12)
        tmp_2 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, 0.1, False, False)
        tmp_3 = None
        return (tmp_4,)