import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.nn.functional.gelu(in_1)
        tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.1, training=False)
        tmp_4 = None
        tmp_6 = torch.nn.functional.linear(tmp_5, tmp_3, tmp_2)
        tmp_5 = tmp_3 = tmp_2 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, p=0.1, training=False)
        tmp_6 = None
        tmp_8 = in_0 + tmp_7
        tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (768,), tmp_1, tmp_0, 1e-05)
        tmp_8 = tmp_1 = tmp_0 = None
        return (tmp_9,)