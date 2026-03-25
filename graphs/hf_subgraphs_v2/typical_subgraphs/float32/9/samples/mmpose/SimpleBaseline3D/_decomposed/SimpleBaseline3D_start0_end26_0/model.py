import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26):
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
        tmp_13 = w_12
        tmp_14 = w_13
        tmp_15 = w_14
        tmp_16 = w_15
        tmp_17 = w_16
        tmp_18 = w_17
        tmp_19 = w_18
        tmp_20 = w_19
        tmp_21 = w_20
        tmp_22 = w_21
        tmp_23 = w_22
        tmp_24 = w_23
        tmp_25 = w_24
        tmp_26 = w_25
        tmp_27 = w_26
        tmp_28 = torch.conv1d(tmp_0, tmp_5, None, (1,), (0,), (1,), 1)
        tmp_0 = tmp_5 = None
        tmp_29 = torch.nn.functional.batch_norm(tmp_28, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 1e-05)
        tmp_28 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        tmp_30 = torch.nn.functional.relu(tmp_29, inplace=True)
        tmp_29 = None
        tmp_31 = torch.nn.functional.dropout(tmp_30, 0.5, False, False)
        tmp_30 = None
        tmp_32 = torch.conv1d(tmp_31, tmp_10, None, (1,), (0,), (1,), 1)
        tmp_10 = None
        tmp_33 = torch.nn.functional.batch_norm(tmp_32, tmp_6, tmp_7, tmp_9, tmp_8, False, 0.1, 1e-05)
        tmp_32 = tmp_6 = tmp_7 = tmp_9 = tmp_8 = None
        tmp_34 = torch.nn.functional.relu(tmp_33, inplace=True)
        tmp_33 = None
        tmp_35 = torch.nn.functional.dropout(tmp_34, 0.5, False, False)
        tmp_34 = None
        tmp_36 = torch.conv1d(tmp_35, tmp_15, None, (1,), (0,), (1,), 1)
        tmp_35 = tmp_15 = None
        tmp_37 = torch.nn.functional.batch_norm(tmp_36, tmp_11, tmp_12, tmp_14, tmp_13, False, 0.1, 1e-05)
        tmp_36 = tmp_11 = tmp_12 = tmp_14 = tmp_13 = None
        tmp_38 = torch.nn.functional.relu(tmp_37, inplace=True)
        tmp_37 = None
        tmp_39 = torch.nn.functional.dropout(tmp_38, 0.5, False, False)
        tmp_38 = None
        tmp_40 = tmp_31[slice(None, None, None), slice(None, None, None), slice(0, 1, None)]
        tmp_31 = None
        tmp_41 = tmp_39 + tmp_40
        tmp_39 = tmp_40 = None
        tmp_42 = torch.conv1d(tmp_41, tmp_20, None, (1,), (0,), (1,), 1)
        tmp_20 = None
        tmp_43 = torch.nn.functional.batch_norm(tmp_42, tmp_16, tmp_17, tmp_19, tmp_18, False, 0.1, 1e-05)
        tmp_42 = tmp_16 = tmp_17 = tmp_19 = tmp_18 = None
        tmp_44 = torch.nn.functional.relu(tmp_43, inplace=True)
        tmp_43 = None
        tmp_45 = torch.nn.functional.dropout(tmp_44, 0.5, False, False)
        tmp_44 = None
        tmp_46 = torch.conv1d(tmp_45, tmp_25, None, (1,), (0,), (1,), 1)
        tmp_45 = tmp_25 = None
        tmp_47 = torch.nn.functional.batch_norm(tmp_46, tmp_21, tmp_22, tmp_24, tmp_23, False, 0.1, 1e-05)
        tmp_46 = tmp_21 = tmp_22 = tmp_24 = tmp_23 = None
        tmp_48 = torch.nn.functional.relu(tmp_47, inplace=True)
        tmp_47 = None
        tmp_49 = torch.nn.functional.dropout(tmp_48, 0.5, False, False)
        tmp_48 = None
        tmp_50 = tmp_41[slice(None, None, None), slice(None, None, None), slice(0, 1, None)]
        tmp_41 = None
        tmp_51 = tmp_49 + tmp_50
        tmp_49 = tmp_50 = None
        tmp_52 = torch.conv1d(tmp_51, tmp_27, tmp_26, (1,), (0,), (1,), 1)
        tmp_51 = tmp_27 = tmp_26 = None
        tmp_53 = tmp_52.reshape(-1, 16, 3)
        tmp_52 = None
        return (tmp_53,)