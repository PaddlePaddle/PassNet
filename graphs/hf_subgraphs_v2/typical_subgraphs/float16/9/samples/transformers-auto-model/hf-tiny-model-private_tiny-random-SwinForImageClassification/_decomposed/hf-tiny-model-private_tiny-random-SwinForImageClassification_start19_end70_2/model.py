import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, in_0, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_1, w_4, w_3);  in_1 = w_4 = w_3 = None
        tmp_18 = linear.view((64, 4, -1, 8));  linear = None
        tmp_19 = tmp_18.transpose(1, 2);  tmp_18 = None
        tmp_20 = in_2.transpose(-1, -2);  in_2 = None
        matmul = torch.matmul(in_3, tmp_20);  in_3 = tmp_20 = None
        tmp_22 = matmul / 2.8284271247461903;  matmul = None
        tmp_23 = w_2.view(-1);  w_2 = None
        tmp_24 = w_5[tmp_23];  w_5 = tmp_23 = None
        tmp_25 = tmp_24.view(4, 4, -1);  tmp_24 = None
        tmp_26 = tmp_25.permute(2, 0, 1);  tmp_25 = None
        tmp_27 = tmp_26.contiguous();  tmp_26 = None
        tmp_28 = tmp_27.unsqueeze(0);  tmp_27 = None
        tmp_29 = tmp_22 + tmp_28;  tmp_22 = tmp_28 = None
        tmp_30 = torch.nn.functional.softmax(tmp_29, dim = -1);  tmp_29 = None
        tmp_31 = torch.nn.functional.dropout(tmp_30, 0.0, False, False);  tmp_30 = None
        matmul_1 = torch.matmul(tmp_31, tmp_19);  tmp_31 = tmp_19 = None
        tmp_33 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_34 = tmp_33.contiguous();  tmp_33 = None
        tmp_35 = tmp_34.view((64, 4, 16));  tmp_34 = None
        linear_1 = torch.nn.functional.linear(tmp_35, w_1, w_0);  tmp_35 = w_1 = w_0 = None
        tmp_37 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_38 = tmp_37.view(-1, 2, 2, 16);  tmp_37 = None
        tmp_39 = tmp_38.view(-1, 8, 8, 2, 2, 16);  tmp_38 = None
        tmp_40 = tmp_39.permute(0, 1, 3, 2, 4, 5);  tmp_39 = None
        tmp_41 = tmp_40.contiguous();  tmp_40 = None
        tmp_42 = tmp_41.view(-1, 16, 16, 16);  tmp_41 = None
        tmp_43 = tmp_42.view(1, 256, 16);  tmp_42 = None
        tmp_44 = in_0 + tmp_43;  in_0 = tmp_43 = None
        tmp_45 = torch.nn.functional.layer_norm(tmp_44, (16,), w_9, w_8, 1e-05);  w_9 = w_8 = None
        linear_2 = torch.nn.functional.linear(tmp_45, w_7, w_6);  tmp_45 = w_7 = w_6 = None
        tmp_47 = torch.nn.functional.gelu(linear_2);  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_47, w_11, w_10);  tmp_47 = w_11 = w_10 = None
        tmp_49 = torch.nn.functional.dropout(linear_3, 0.0, False, False);  linear_3 = None
        tmp_50 = tmp_44 + tmp_49;  tmp_44 = tmp_49 = None
        tmp_51 = tmp_50.view(1, 16, 16, 16);  tmp_50 = None
        tmp_52 = tmp_51[(slice(None, None, None), slice(0, None, 2), slice(0, None, 2), slice(None, None, None))]
        tmp_53 = tmp_51[(slice(None, None, None), slice(1, None, 2), slice(0, None, 2), slice(None, None, None))]
        tmp_54 = tmp_51[(slice(None, None, None), slice(0, None, 2), slice(1, None, 2), slice(None, None, None))]
        tmp_55 = tmp_51[(slice(None, None, None), slice(1, None, 2), slice(1, None, 2), slice(None, None, None))];  tmp_51 = None
        tmp_56 = torch.cat([tmp_52, tmp_53, tmp_54, tmp_55], -1);  tmp_52 = tmp_53 = tmp_54 = tmp_55 = None
        tmp_57 = tmp_56.view(1, -1, 64);  tmp_56 = None
        tmp_58 = torch.nn.functional.layer_norm(tmp_57, (64,), w_13, w_12, 1e-05);  tmp_57 = w_13 = w_12 = None
        linear_4 = torch.nn.functional.linear(tmp_58, w_14, None);  tmp_58 = w_14 = None
        tmp_60 = torch.nn.functional.layer_norm(linear_4, (32,), w_16, w_15, 1e-05);  w_16 = w_15 = None
        tmp_61 = tmp_60.view(1, 8, 8, 32);  tmp_60 = None
        tmp_62 = torch.nn.functional.pad(tmp_61, (0, 0, 0, 0, 0, 0), 'constant', None);  tmp_61 = None
        tmp_63 = tmp_62.view(1, 4, 2, 4, 2, 32);  tmp_62 = None
        tmp_64 = tmp_63.permute(0, 1, 3, 2, 4, 5);  tmp_63 = None
        tmp_65 = tmp_64.contiguous();  tmp_64 = None
        tmp_66 = tmp_65.view(-1, 2, 2, 32);  tmp_65 = None
        tmp_67 = tmp_66.view(-1, 4, 32);  tmp_66 = None
        return (tmp_67, linear_4)
        