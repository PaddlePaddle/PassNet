import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.nn.functional.gelu(in_1)
        tmp_5 = torch.nn.functional.linear(tmp_4, tmp_3, tmp_2)
        tmp_4 = tmp_3 = tmp_2 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.1, False, False)
        tmp_5 = None
        tmp_7 = tmp_6 + in_0
        tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (512,), tmp_1, tmp_0, 1e-12)
        tmp_7 = tmp_1 = tmp_0 = None
        return (tmp_8,)