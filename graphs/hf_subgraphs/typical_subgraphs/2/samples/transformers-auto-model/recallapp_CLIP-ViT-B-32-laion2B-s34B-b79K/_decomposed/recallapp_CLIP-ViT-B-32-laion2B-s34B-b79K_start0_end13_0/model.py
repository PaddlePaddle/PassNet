import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14):
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
        tmp_13 = in_13
        tmp_14 = in_14
        tmp_15 = tmp_0.to(dtype=torch.float32)
        tmp_0 = None
        tmp_16 = torch.conv2d(tmp_15, tmp_2, None, (32, 32), (0, 0), (1, 1), 1)
        tmp_15 = tmp_2 = None
        tmp_17 = tmp_16.flatten(2)
        tmp_16 = None
        tmp_18 = tmp_17.transpose(1, 2)
        tmp_17 = None
        tmp_19 = tmp_4.expand(1, 1, -1)
        tmp_4 = None
        tmp_20 = torch.cat([tmp_19, tmp_18], dim=1)
        tmp_19 = tmp_18 = None
        tmp_21 = torch.nn.functional.embedding(tmp_1, tmp_3, None, None, 2.0, False, False)
        tmp_1 = tmp_3 = None
        tmp_22 = tmp_20 + tmp_21
        tmp_20 = tmp_21 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (768,), tmp_14, tmp_13, 1e-05)
        tmp_22 = tmp_14 = tmp_13 = None
        tmp_24 = torch.nn.functional.layer_norm(tmp_23, (768,), tmp_6, tmp_5, 1e-05)
        tmp_6 = tmp_5 = None
        tmp_25 = torch.nn.functional.linear(tmp_24, tmp_10, tmp_9)
        tmp_10 = tmp_9 = None
        tmp_26 = torch.nn.functional.linear(tmp_24, tmp_8, tmp_7)
        tmp_8 = tmp_7 = None
        tmp_27 = torch.nn.functional.linear(tmp_24, tmp_12, tmp_11)
        tmp_24 = tmp_12 = tmp_11 = None
        return (tmp_23, tmp_26, tmp_25, tmp_27)