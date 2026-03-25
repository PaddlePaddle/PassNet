import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = torch.nn.functional.gelu(in_5)
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.1, False, False)
        tmp_9 = None
        tmp_11 = torch.nn.functional.linear(tmp_10, tmp_3, tmp_2)
        tmp_10 = tmp_3 = tmp_2 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.1, False, False)
        tmp_11 = None
        tmp_13 = in_4 + tmp_12
        tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (768,), tmp_5, tmp_4, 1e-05)
        tmp_13 = tmp_5 = tmp_4 = None
        tmp_15 = torch.stack((in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_0, in_1, in_2, in_3, tmp_14), dim=1)
        tmp_16 = torch.nn.functional.softmax(tmp_8, dim=-1)
        tmp_8 = None
        tmp_17 = tmp_16.view(-1, 1, 1)
        tmp_16 = None
        tmp_18 = tmp_15 * tmp_17
        tmp_15 = tmp_17 = None
        tmp_19 = tmp_18.sum(dim=1)
        tmp_18 = None
        tmp_20 = torch.nn.functional.linear(tmp_19, tmp_7, tmp_6)
        tmp_19 = tmp_7 = tmp_6 = None
        tmp_21 = tmp_20.mean(dim=1)
        tmp_20 = None
        tmp_22 = torch.nn.functional.linear(tmp_21, tmp_1, tmp_0)
        tmp_21 = tmp_1 = tmp_0 = None
        return (tmp_14, tmp_22)