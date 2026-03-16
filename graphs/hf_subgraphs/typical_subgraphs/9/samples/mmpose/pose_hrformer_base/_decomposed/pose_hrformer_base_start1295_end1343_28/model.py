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
        tmp_11 = torch.nn.functional.gelu(in_0, approximate='none')
        tmp_12 = tmp_11.flatten(2)
        tmp_11 = None
        tmp_13 = tmp_12.transpose(1, 2)
        tmp_12 = None
        tmp_14 = tmp_13.contiguous()
        tmp_13 = None
        tmp_15 = in_1 + tmp_14
        tmp_14 = None
        tmp_16 = tmp_15.permute(0, 2, 1)
        tmp_15 = None
        tmp_17 = tmp_16.view(1, 312, 16, 12)
        tmp_16 = None
        tmp_18 = tmp_17.view(1, 312, -1)
        tmp_17 = None
        tmp_19 = tmp_18.permute(0, 2, 1)
        tmp_18 = None
        tmp_20 = torch.nn.functional.layer_norm(tmp_19, (312,), tmp_8, tmp_7, 1e-06)
        tmp_8 = tmp_7 = None
        tmp_21 = tmp_20.view(1, 16, 12, 312)
        tmp_20 = None
        tmp_22 = torch.nn.functional.pad(tmp_21, (0, 0, 1, 1, 2, 3), 'constant', None)
        tmp_21 = None
        tmp_23 = tmp_22.view(1, 3, 7, 2, 7, 312)
        tmp_22 = None
        tmp_24 = tmp_23.permute(0, 1, 3, 2, 4, 5)
        tmp_23 = None
        tmp_25 = tmp_24.reshape(-1, 49, 312)
        tmp_24 = None
        tmp_26 = torch.nn.functional.linear(tmp_25, tmp_4, tmp_3)
        tmp_25 = tmp_4 = tmp_3 = None
        tmp_27 = tmp_26.reshape(6, 49, 3, 8, 39)
        tmp_26 = None
        tmp_28 = tmp_27.permute(2, 0, 3, 1, 4)
        tmp_27 = None
        tmp_29 = tmp_28[0]
        tmp_30 = tmp_28[1]
        tmp_31 = tmp_28[2]
        tmp_28 = None
        tmp_32 = tmp_6.item()
        tmp_6 = None
        tmp_33 = tmp_29 * tmp_32
        tmp_29 = tmp_32 = None
        tmp_34 = tmp_30.transpose(-2, -1)
        tmp_30 = None
        tmp_35 = tmp_33 @ tmp_34
        tmp_33 = tmp_34 = None
        tmp_36 = tmp_0.view(-1)
        tmp_0 = None
        tmp_37 = tmp_5[tmp_36]
        tmp_5 = tmp_36 = None
        tmp_38 = tmp_37.view(49, 49, -1)
        tmp_37 = None
        tmp_39 = tmp_38.permute(2, 0, 1)
        tmp_38 = None
        tmp_40 = tmp_39.contiguous()
        tmp_39 = None
        tmp_41 = tmp_40.unsqueeze(0)
        tmp_40 = None
        tmp_42 = tmp_35 + tmp_41
        tmp_35 = tmp_41 = None
        tmp_43 = torch.nn.functional.softmax(tmp_42, -1, _stacklevel=5)
        tmp_42 = None
        tmp_44 = torch.nn.functional.dropout(tmp_43, 0.0, False, False)
        tmp_43 = None
        tmp_45 = tmp_44 @ tmp_31
        tmp_44 = tmp_31 = None
        tmp_46 = tmp_45.transpose(1, 2)
        tmp_45 = None
        tmp_47 = tmp_46.reshape(6, 49, 312)
        tmp_46 = None
        tmp_48 = torch.nn.functional.linear(tmp_47, tmp_2, tmp_1)
        tmp_47 = tmp_2 = tmp_1 = None
        tmp_49 = torch.nn.functional.dropout(tmp_48, 0.0, False, False)
        tmp_48 = None
        tmp_50 = tmp_49.reshape(1, 3, 2, 7, 7, 312)
        tmp_49 = None
        tmp_51 = tmp_50.permute(0, 1, 3, 2, 4, 5)
        tmp_50 = None
        tmp_52 = tmp_51.reshape(1, 21, 14, 312)
        tmp_51 = None
        tmp_53 = tmp_52[slice(None, None, None), slice(2, 18, None), slice(1, 13, None)]
        tmp_52 = None
        tmp_54 = tmp_53.reshape(1, 192, 312)
        tmp_53 = None
        tmp_55 = tmp_19 + tmp_54
        tmp_19 = tmp_54 = None
        tmp_56 = torch.nn.functional.layer_norm(tmp_55, (312,), tmp_10, tmp_9, 1e-06)
        tmp_10 = tmp_9 = None
        tmp_57 = tmp_56.transpose(1, 2)
        tmp_56 = None
        tmp_58 = tmp_57.reshape(1, 312, 16, 12)
        tmp_57 = None
        return (tmp_55, tmp_58)