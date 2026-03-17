import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = w_9
        tmp_10 = w_10
        tmp_11 = torch.nn.functional.linear(in_0, tmp_10, tmp_9)
        tmp_10 = tmp_9 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, p=0.1, training=False)
        tmp_11 = None
        tmp_13 = in_1 + tmp_12
        tmp_12 = None
        tmp_14 = tmp_13.reshape(-1, 768)
        tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (768,), tmp_8, tmp_7, 1e-05)
        tmp_8 = tmp_7 = None
        tmp_16 = torch.nn.functional.linear(tmp_15, tmp_4, tmp_3)
        tmp_15 = tmp_4 = tmp_3 = None
        tmp_17 = torch.nn.functional.relu(tmp_16, inplace=False)
        tmp_16 = None
        tmp_18 = torch.nn.functional.linear(tmp_17, tmp_6, tmp_5)
        tmp_17 = tmp_6 = tmp_5 = None
        tmp_19 = torch.nn.functional.dropout(tmp_18, p=0.1, training=False)
        tmp_18 = None
        tmp_20 = tmp_14 + tmp_19
        tmp_14 = tmp_19 = None
        tmp_21 = tmp_20.view((1, 20, 768))
        tmp_20 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (768,), tmp_2, tmp_1, 1e-05)
        tmp_21 = tmp_2 = tmp_1 = None
        tmp_23 = torch.nn.functional.linear(tmp_22, tmp_0, None)
        tmp_22 = tmp_0 = None
        tmp_24 = tmp_23.contiguous()
        tmp_23 = None
        return (tmp_24,)