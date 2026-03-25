import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.nn.functional.linear(in_1, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.1, False, False)
        tmp_4 = None
        tmp_6 = tmp_5 + in_0
        tmp_5 = None
        tmp_7 = torch.nn.functional.layer_norm(tmp_6, (1024,), tmp_1, tmp_0, 1e-12)
        tmp_6 = tmp_1 = tmp_0 = None
        return (tmp_7,)