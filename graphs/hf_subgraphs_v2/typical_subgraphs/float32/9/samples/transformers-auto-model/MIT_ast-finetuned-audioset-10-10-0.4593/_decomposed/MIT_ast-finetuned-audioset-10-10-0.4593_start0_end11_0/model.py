import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = w_6
        tmp_8 = tmp_0.unsqueeze(1)
        tmp_0 = None
        tmp_9 = tmp_8.transpose(2, 3)
        tmp_8 = None
        tmp_10 = torch.conv2d(tmp_9, tmp_2, tmp_1, (10, 10), (0, 0), (1, 1), 1)
        tmp_9 = tmp_2 = tmp_1 = None
        tmp_11 = tmp_10.flatten(2)
        tmp_10 = None
        tmp_12 = tmp_11.transpose(1, 2)
        tmp_11 = None
        tmp_13 = tmp_3.expand(1, -1, -1)
        tmp_3 = None
        tmp_14 = tmp_4.expand(1, -1, -1)
        tmp_4 = None
        tmp_15 = torch.cat((tmp_13, tmp_14, tmp_12), dim=1)
        tmp_13 = tmp_14 = tmp_12 = None
        tmp_16 = tmp_15 + tmp_5
        tmp_15 = tmp_5 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.0, False, False)
        tmp_16 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (768,), tmp_7, tmp_6, 1e-12)
        tmp_7 = tmp_6 = None
        return (tmp_17, tmp_18)