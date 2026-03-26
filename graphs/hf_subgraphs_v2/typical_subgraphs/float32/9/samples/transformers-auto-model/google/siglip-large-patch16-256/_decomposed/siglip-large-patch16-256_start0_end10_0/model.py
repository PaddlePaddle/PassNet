import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = w_6
        tmp_8 = w_7
        tmp_9 = w_8
        tmp_10 = w_9
        tmp_11 = w_10
        tmp_12 = w_11
        tmp_13 = tmp_0.to(dtype=torch.float32)
        tmp_0 = None
        tmp_14 = torch.conv2d(tmp_13, tmp_3, tmp_2, (16, 16), 'valid', (1, 1), 1)
        tmp_13 = tmp_3 = tmp_2 = None
        tmp_15 = tmp_14.flatten(2)
        tmp_14 = None
        tmp_16 = tmp_15.transpose(1, 2)
        tmp_15 = None
        tmp_17 = torch.nn.functional.embedding(tmp_1, tmp_4, None, None, 2.0, False, False)
        tmp_1 = tmp_4 = None
        tmp_18 = tmp_16 + tmp_17
        tmp_16 = tmp_17 = None
        tmp_19 = torch.nn.functional.layer_norm(tmp_18, (1024,), tmp_6, tmp_5, 1e-06)
        tmp_6 = tmp_5 = None
        tmp_20 = torch.nn.functional.linear(tmp_19, tmp_10, tmp_9)
        tmp_10 = tmp_9 = None
        tmp_21 = torch.nn.functional.linear(tmp_19, tmp_8, tmp_7)
        tmp_8 = tmp_7 = None
        tmp_22 = torch.nn.functional.linear(tmp_19, tmp_12, tmp_11)
        tmp_19 = tmp_12 = tmp_11 = None
        return (tmp_18, tmp_21, tmp_20, tmp_22)