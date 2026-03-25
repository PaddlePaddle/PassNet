import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = tmp_0.to(dtype=torch.float32)
        tmp_0 = None
        tmp_8 = torch.conv2d(tmp_7, tmp_2, tmp_1, (16, 16), (0, 0), (1, 1), 1)
        tmp_7 = tmp_2 = tmp_1 = None
        tmp_9 = tmp_8.flatten(2)
        tmp_8 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = tmp_3.expand(1, -1, -1)
        tmp_3 = None
        tmp_12 = torch.cat((tmp_11, tmp_10), dim=1)
        tmp_11 = tmp_10 = None
        tmp_13 = tmp_12 + tmp_4
        tmp_12 = tmp_4 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False)
        tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (1024,), tmp_6, tmp_5, 1e-06)
        tmp_6 = tmp_5 = None
        return (tmp_14, tmp_15)