import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.nn.functional.linear(in_1, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.1, training=False)
        tmp_4 = None
        tmp_6 = in_0 + tmp_5
        tmp_5 = None
        tmp_7 = torch.nn.functional.layer_norm(tmp_6, (512,), tmp_3, tmp_2, 1e-05)
        tmp_6 = tmp_3 = tmp_2 = None
        return (tmp_7,)