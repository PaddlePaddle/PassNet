import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = torch.nn.functional.relu(in_7, inplace=False)
        tmp_7 = torch.nn.functional.dropout(tmp_6, p=0.0, training=False)
        tmp_6 = None
        tmp_8 = torch.nn.functional.linear(tmp_7, tmp_3, tmp_2)
        tmp_7 = tmp_3 = tmp_2 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, p=0.1, training=False)
        tmp_8 = None
        tmp_10 = in_6 + tmp_9
        tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (256,), tmp_5, tmp_4, 1e-05)
        tmp_10 = tmp_5 = tmp_4 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (256,), tmp_1, tmp_0, 1e-05)
        tmp_11 = tmp_1 = tmp_0 = None
        return (tmp_12,)