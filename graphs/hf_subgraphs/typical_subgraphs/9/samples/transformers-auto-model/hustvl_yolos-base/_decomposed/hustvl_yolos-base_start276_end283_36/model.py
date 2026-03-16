import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.nn.functional.gelu(in_1)
        tmp_5 = torch.nn.functional.linear(tmp_4, tmp_1, tmp_0)
        tmp_4 = tmp_1 = tmp_0 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
        tmp_5 = None
        tmp_7 = tmp_6 + in_0
        tmp_6 = None
        tmp_8 = in_2[8]
        tmp_9 = tmp_7 + tmp_8
        tmp_7 = tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (768,), tmp_3, tmp_2, 1e-12)
        tmp_3 = tmp_2 = None
        return (tmp_9, tmp_10)