import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = torch.nn.functional.relu(in_1, inplace=False)
        tmp_7 = torch.nn.functional.dropout(tmp_6, p=0.0, training=False)
        tmp_6 = None
        tmp_8 = torch.nn.functional.linear(tmp_7, tmp_5, tmp_4)
        tmp_7 = tmp_5 = tmp_4 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, p=0.1, training=False)
        tmp_8 = None
        tmp_10 = in_0 + tmp_9
        tmp_9 = None
        tmp_11 = torch.rand([])
        tmp_11 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_10, (1024,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        tmp_13 = torch.nn.functional.linear(tmp_12, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        return (tmp_10, tmp_12, tmp_13)