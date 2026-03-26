import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15):
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
        tmp_12 = in_13.transpose(2, 3)
        tmp_13 = torch.matmul(in_14, tmp_12)
        tmp_12 = None
        tmp_14 = tmp_13 * 1.0
        tmp_13 = None
        tmp_15 = torch.nn.functional.softmax(tmp_14, dim=-1, dtype=torch.float32)
        tmp_14 = None
        tmp_16 = tmp_15.to(torch.float32)
        tmp_15 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, p=0.0, training=False)
        tmp_16 = None
        tmp_18 = torch.matmul(tmp_17, in_15)
        tmp_17 = None
        tmp_19 = tmp_18.transpose(1, 2)
        tmp_18 = None
        tmp_20 = tmp_19.contiguous()
        tmp_19 = None
        tmp_21 = tmp_20.reshape(1, 257, -1)
        tmp_20 = None
        tmp_22 = tmp_21.contiguous()
        tmp_21 = None
        tmp_23 = torch.nn.functional.linear(tmp_22, tmp_7, tmp_6)
        tmp_22 = tmp_7 = tmp_6 = None
        tmp_24 = in_12 + tmp_23
        tmp_23 = None
        tmp_25 = torch.nn.functional.layer_norm(tmp_24, (1024,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        tmp_26 = torch.nn.functional.linear(tmp_25, tmp_3, tmp_2)
        tmp_25 = tmp_3 = tmp_2 = None
        tmp_27 = 1.702 * tmp_26
        tmp_28 = torch.sigmoid(tmp_27)
        tmp_27 = None
        tmp_29 = tmp_26 * tmp_28
        tmp_26 = tmp_28 = None
        tmp_30 = torch.nn.functional.linear(tmp_29, tmp_5, tmp_4)
        tmp_29 = tmp_5 = tmp_4 = None
        tmp_31 = tmp_24 + tmp_30
        tmp_24 = tmp_30 = None
        tmp_32 = torch.nn.functional.layer_norm(tmp_31, (1024,), tmp_9, tmp_8, 1e-05)
        tmp_9 = tmp_8 = None
        tmp_33 = torch.nn.functional.linear(tmp_32, tmp_11, tmp_10)
        tmp_11 = tmp_10 = None
        tmp_34 = tmp_33.view((1, 257, -1, 64))
        tmp_33 = None
        tmp_35 = tmp_34.transpose(1, 2)
        tmp_34 = None
        tmp_36 = tmp_35 * 0.125
        tmp_35 = None
        return (tmp_31, tmp_32, tmp_36)