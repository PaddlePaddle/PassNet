import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0, in_1):
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
        tmp_12 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_13 = tmp_12.view(1, 512, -1)
        tmp_12 = None
        tmp_14 = tmp_13.transpose(1, 2)
        tmp_13 = None
        tmp_15 = tmp_14.contiguous()
        tmp_14 = None
        tmp_16 = tmp_3.view((1, 1, 32))
        tmp_3 = None
        tmp_17 = tmp_15.unsqueeze(2)
        tmp_18 = tmp_17.expand((1, 4096, 32, 512))
        tmp_17 = None
        tmp_19 = tmp_2.view((1, 1, 32, 512))
        tmp_20 = tmp_18 - tmp_19
        tmp_18 = tmp_19 = None
        tmp_21 = tmp_20.pow(2)
        tmp_20 = None
        tmp_22 = tmp_21.sum(dim=3)
        tmp_21 = None
        tmp_23 = tmp_16 * tmp_22
        tmp_16 = tmp_22 = None
        tmp_24 = torch.nn.functional.softmax(tmp_23, dim=2)
        tmp_23 = None
        tmp_25 = tmp_2.view((1, 1, 32, 512))
        tmp_2 = None
        tmp_26 = tmp_15.unsqueeze(2)
        tmp_15 = None
        tmp_27 = tmp_26.expand((1, 4096, 32, 512))
        tmp_26 = None
        tmp_28 = tmp_24.unsqueeze(3)
        tmp_24 = None
        tmp_29 = tmp_27 - tmp_25
        tmp_27 = tmp_25 = None
        tmp_30 = tmp_28 * tmp_29
        tmp_28 = tmp_29 = None
        tmp_31 = tmp_30.sum(dim=1)
        tmp_30 = None
        tmp_32 = torch.nn.functional.batch_norm(tmp_31, tmp_4, tmp_5, tmp_7, tmp_6, False, 0.1, 1e-05)
        tmp_31 = tmp_4 = tmp_5 = tmp_7 = tmp_6 = None
        tmp_33 = torch.nn.functional.relu(tmp_32, inplace=True)
        tmp_32 = None
        tmp_34 = tmp_33.mean(dim=1)
        tmp_33 = None
        tmp_35 = torch.nn.functional.linear(tmp_34, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_36 = torch.sigmoid(tmp_35)
        tmp_35 = None
        tmp_37 = tmp_36.view(1, 512, 1, 1)
        tmp_36 = None
        tmp_38 = in_0 * tmp_37
        tmp_37 = None
        tmp_39 = in_0 + tmp_38
        tmp_38 = None
        tmp_40 = torch.relu_(tmp_39)
        tmp_39 = None
        tmp_41 = torch.nn.functional.dropout2d(tmp_40, 0.1, False, False)
        tmp_40 = None
        tmp_42 = torch.conv2d(tmp_41, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_41 = tmp_1 = tmp_0 = None
        tmp_43 = torch.nn.functional.linear(tmp_34, tmp_11, tmp_10)
        tmp_34 = tmp_11 = tmp_10 = None
        return (tmp_42, tmp_43)