import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0, in_1, in_2, in_3):
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
        tmp_11 = w_11
        tmp_12 = in_1.transpose(2, 3)
        tmp_13 = torch.matmul(in_2, tmp_12)
        tmp_12 = None
        tmp_14 = tmp_13 * 1.0
        tmp_13 = None
        tmp_15 = torch.nn.functional.softmax(tmp_14, dim=-1, dtype=torch.float32)
        tmp_14 = None
        tmp_16 = tmp_15.to(torch.float32)
        tmp_15 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, p=0.0, training=False)
        tmp_16 = None
        tmp_18 = torch.matmul(tmp_17, in_3)
        tmp_17 = None
        tmp_19 = tmp_18.transpose(1, 2)
        tmp_18 = None
        tmp_20 = tmp_19.contiguous()
        tmp_19 = None
        tmp_21 = tmp_20.reshape(1, 257, -1)
        tmp_20 = None
        tmp_22 = tmp_21.contiguous()
        tmp_21 = None
        tmp_23 = torch.nn.functional.linear(tmp_22, tmp_11, tmp_10)
        tmp_22 = tmp_11 = tmp_10 = None
        tmp_24 = in_0 + tmp_23
        tmp_23 = None
        tmp_25 = torch.nn.functional.layer_norm(tmp_24, (1280,), tmp_5, tmp_4, 1e-05)
        tmp_5 = tmp_4 = None
        tmp_26 = torch.nn.functional.linear(tmp_25, tmp_7, tmp_6)
        tmp_25 = tmp_7 = tmp_6 = None
        tmp_27 = 1.702 * tmp_26
        tmp_28 = torch.sigmoid(tmp_27)
        tmp_27 = None
        tmp_29 = tmp_26 * tmp_28
        tmp_26 = tmp_28 = None
        tmp_30 = torch.nn.functional.linear(tmp_29, tmp_9, tmp_8)
        tmp_29 = tmp_9 = tmp_8 = None
        tmp_31 = tmp_24 + tmp_30
        tmp_24 = tmp_30 = None
        tmp_32 = torch.nn.functional.layer_norm(tmp_31, (1280,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        tmp_33 = torch.nn.functional.linear(tmp_32, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_34 = tmp_33.view((1, 257, -1, 80))
        tmp_33 = None
        tmp_35 = tmp_34.transpose(1, 2)
        tmp_34 = None
        tmp_36 = tmp_35 * 0.11180339887498948
        tmp_35 = None
        return (tmp_31, tmp_32, tmp_36)