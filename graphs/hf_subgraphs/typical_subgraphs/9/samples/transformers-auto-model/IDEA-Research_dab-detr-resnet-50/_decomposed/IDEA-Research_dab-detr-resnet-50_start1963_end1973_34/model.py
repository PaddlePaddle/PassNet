import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = torch.prelu(in_6, tmp_2)
        tmp_2 = None
        tmp_8 = torch.nn.functional.dropout(tmp_7, p=0.0, training=False)
        tmp_7 = None
        tmp_9 = torch.nn.functional.linear(tmp_8, tmp_4, tmp_3)
        tmp_8 = tmp_4 = tmp_3 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, p=0.1, training=False)
        tmp_9 = None
        tmp_11 = in_0 + tmp_10
        tmp_10 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (256,), tmp_6, tmp_5, 1e-05)
        tmp_11 = tmp_6 = tmp_5 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (256,), tmp_1, tmp_0, 1e-05)
        tmp_14 = torch.nn.functional.layer_norm(tmp_12, (256,), tmp_1, tmp_0, 1e-05)
        tmp_12 = tmp_1 = tmp_0 = None
        tmp_15 = torch.stack([in_1, in_2, in_3, in_4, in_5, tmp_13])
        tmp_13 = None
        tmp_16 = torch.stack([in_7])
        return (tmp_14, tmp_15, tmp_16)