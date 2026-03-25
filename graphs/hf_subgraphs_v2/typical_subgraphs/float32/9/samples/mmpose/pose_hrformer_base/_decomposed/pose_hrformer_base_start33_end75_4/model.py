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
        tmp_11 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_12 = in_0.view(1, 78, -1)
        tmp_13 = tmp_12.permute(0, 2, 1)
        tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (78,), tmp_8, tmp_7, 1e-06)
        tmp_8 = tmp_7 = None
        tmp_15 = tmp_14.view(1, 64, 48, 78)
        tmp_14 = None
        tmp_16 = torch.nn.functional.pad(tmp_15, (0, 0, 0, 1, 3, 3), 'constant', None)
        tmp_15 = None
        tmp_17 = tmp_16.view(1, 10, 7, 7, 7, 78)
        tmp_16 = None
        tmp_18 = tmp_17.permute(0, 1, 3, 2, 4, 5)
        tmp_17 = None
        tmp_19 = tmp_18.reshape(-1, 49, 78)
        tmp_18 = None
        tmp_20 = torch.nn.functional.linear(tmp_19, tmp_4, tmp_3)
        tmp_19 = tmp_4 = tmp_3 = None
        tmp_21 = tmp_20.reshape(70, 49, 3, 2, 39)
        tmp_20 = None
        tmp_22 = tmp_21.permute(2, 0, 3, 1, 4)
        tmp_21 = None
        tmp_23 = tmp_22[0]
        tmp_24 = tmp_22[1]
        tmp_25 = tmp_22[2]
        tmp_22 = None
        tmp_26 = tmp_6.item()
        tmp_6 = None
        tmp_27 = tmp_23 * tmp_26
        tmp_23 = tmp_26 = None
        tmp_28 = tmp_24.transpose(-2, -1)
        tmp_24 = None
        tmp_29 = tmp_27 @ tmp_28
        tmp_27 = tmp_28 = None
        tmp_30 = tmp_0.view(-1)
        tmp_0 = None
        tmp_31 = tmp_5[tmp_30]
        tmp_5 = tmp_30 = None
        tmp_32 = tmp_31.view(49, 49, -1)
        tmp_31 = None
        tmp_33 = tmp_32.permute(2, 0, 1)
        tmp_32 = None
        tmp_34 = tmp_33.contiguous()
        tmp_33 = None
        tmp_35 = tmp_34.unsqueeze(0)
        tmp_34 = None
        tmp_36 = tmp_29 + tmp_35
        tmp_29 = tmp_35 = None
        tmp_37 = torch.nn.functional.softmax(tmp_36, -1, _stacklevel=5)
        tmp_36 = None
        tmp_38 = torch.nn.functional.dropout(tmp_37, 0.0, False, False)
        tmp_37 = None
        tmp_39 = tmp_38 @ tmp_25
        tmp_38 = tmp_25 = None
        tmp_40 = tmp_39.transpose(1, 2)
        tmp_39 = None
        tmp_41 = tmp_40.reshape(70, 49, 78)
        tmp_40 = None
        tmp_42 = torch.nn.functional.linear(tmp_41, tmp_2, tmp_1)
        tmp_41 = tmp_2 = tmp_1 = None
        tmp_43 = torch.nn.functional.dropout(tmp_42, 0.0, False, False)
        tmp_42 = None
        tmp_44 = tmp_43.reshape(1, 10, 7, 7, 7, 78)
        tmp_43 = None
        tmp_45 = tmp_44.permute(0, 1, 3, 2, 4, 5)
        tmp_44 = None
        tmp_46 = tmp_45.reshape(1, 70, 49, 78)
        tmp_45 = None
        tmp_47 = tmp_46[slice(None, None, None), slice(3, 67, None), slice(0, 48, None)]
        tmp_46 = None
        tmp_48 = tmp_47.reshape(1, 3072, 78)
        tmp_47 = None
        tmp_49 = tmp_13 + tmp_48
        tmp_13 = tmp_48 = None
        tmp_50 = torch.nn.functional.layer_norm(tmp_49, (78,), tmp_10, tmp_9, 1e-06)
        tmp_10 = tmp_9 = None
        tmp_51 = tmp_50.transpose(1, 2)
        tmp_50 = None
        tmp_52 = tmp_51.reshape(1, 78, 64, 48)
        tmp_51 = None
        return (tmp_11, tmp_49, tmp_52)