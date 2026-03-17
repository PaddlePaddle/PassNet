import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = in_8
        tmp_9 = in_9
        tmp_10 = in_10
        tmp_11 = in_11
        tmp_12 = in_12
        tmp_13 = tmp_0.to(dtype=torch.float32)
        tmp_0 = None
        tmp_14 = torch.conv2d(tmp_13, tmp_3, tmp_2, (32, 32), 'valid', (1, 1), 1)
        tmp_13 = tmp_3 = tmp_2 = None
        tmp_15 = tmp_14.flatten(2)
        tmp_14 = None
        tmp_16 = tmp_15.transpose(1, 2)
        tmp_15 = None
        tmp_17 = torch.nn.functional.embedding(tmp_1, tmp_4, None, None, 2.0, False, False)
        tmp_1 = tmp_4 = None
        tmp_18 = tmp_16 + tmp_17
        tmp_16 = tmp_17 = None
        tmp_19 = torch.nn.functional.layer_norm(tmp_18, (768,), tmp_6, tmp_5, 1e-06)
        tmp_6 = tmp_5 = None
        tmp_20 = torch.nn.functional.linear(tmp_19, tmp_10, tmp_9)
        tmp_10 = tmp_9 = None
        tmp_21 = torch.nn.functional.linear(tmp_19, tmp_8, tmp_7)
        tmp_8 = tmp_7 = None
        tmp_22 = torch.nn.functional.linear(tmp_19, tmp_12, tmp_11)
        tmp_19 = tmp_12 = tmp_11 = None
        return (tmp_18, tmp_21, tmp_20, tmp_22)