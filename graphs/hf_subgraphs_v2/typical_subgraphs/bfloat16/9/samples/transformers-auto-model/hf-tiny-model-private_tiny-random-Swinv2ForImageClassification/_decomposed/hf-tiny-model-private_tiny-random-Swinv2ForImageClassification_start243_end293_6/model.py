import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, in_0, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_0, w_10, w_9);  in_0 = w_10 = w_9 = None
        tmp_23 = linear.view(4, -1, 4, 16);  linear = None
        tmp_24 = tmp_23.transpose(1, 2);  tmp_23 = None
        tmp_25 = torch.nn.functional.normalize(in_3, dim = -1);  in_3 = None
        tmp_26 = torch.nn.functional.normalize(in_2, dim = -1);  in_2 = None
        tmp_27 = tmp_26.transpose(-2, -1);  tmp_26 = None
        matmul = tmp_25 @ tmp_27;  tmp_25 = tmp_27 = None
        tmp_29 = torch.clamp(w_11, max = 4.605170185988092);  w_11 = None
        tmp_30 = tmp_29.exp();  tmp_29 = None
        tmp_31 = matmul * tmp_30;  matmul = tmp_30 = None
        linear_1 = torch.nn.functional.linear(w_4, w_7, w_6);  w_4 = w_7 = w_6 = None
        tmp_33 = torch.nn.functional.relu(linear_1, inplace = True);  linear_1 = None
        linear_2 = torch.nn.functional.linear(tmp_33, w_8, None);  tmp_33 = w_8 = None
        tmp_35 = linear_2.view(-1, 4);  linear_2 = None
        tmp_36 = w_5.view(-1);  w_5 = None
        tmp_37 = tmp_35[tmp_36];  tmp_35 = tmp_36 = None
        tmp_38 = tmp_37.view(4, 4, -1);  tmp_37 = None
        tmp_39 = tmp_38.permute(2, 0, 1);  tmp_38 = None
        tmp_40 = tmp_39.contiguous();  tmp_39 = None
        tmp_41 = torch.sigmoid(tmp_40);  tmp_40 = None
        tmp_42 = 16 * tmp_41;  tmp_41 = None
        tmp_43 = tmp_42.unsqueeze(0);  tmp_42 = None
        tmp_44 = tmp_31 + tmp_43;  tmp_31 = tmp_43 = None
        tmp_45 = torch.nn.functional.softmax(tmp_44, dim = -1);  tmp_44 = None
        tmp_46 = torch.nn.functional.dropout(tmp_45, 0.0, False, False);  tmp_45 = None
        matmul_1 = torch.matmul(tmp_46, tmp_24);  tmp_46 = tmp_24 = None
        tmp_48 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_49 = tmp_48.contiguous();  tmp_48 = None
        tmp_50 = tmp_49.view((4, 4, 64));  tmp_49 = None
        linear_3 = torch.nn.functional.linear(tmp_50, w_3, w_2);  tmp_50 = w_3 = w_2 = None
        tmp_52 = torch.nn.functional.dropout(linear_3, 0.0, False, False);  linear_3 = None
        tmp_53 = tmp_52.view(-1, 2, 2, 64);  tmp_52 = None
        tmp_54 = tmp_53.view(-1, 2, 2, 2, 2, 64);  tmp_53 = None
        tmp_55 = tmp_54.permute(0, 1, 3, 2, 4, 5);  tmp_54 = None
        tmp_56 = tmp_55.contiguous();  tmp_55 = None
        tmp_57 = tmp_56.view(-1, 4, 4, 64);  tmp_56 = None
        tmp_58 = tmp_57.view(1, 16, 64);  tmp_57 = None
        tmp_59 = torch.nn.functional.layer_norm(tmp_58, (64,), w_17, w_16, 1e-05);  tmp_58 = w_17 = w_16 = None
        tmp_60 = in_1 + tmp_59;  in_1 = tmp_59 = None
        linear_4 = torch.nn.functional.linear(tmp_60, w_13, w_12);  w_13 = w_12 = None
        tmp_62 = torch.nn.functional.gelu(linear_4);  linear_4 = None
        linear_5 = torch.nn.functional.linear(tmp_62, w_19, w_18);  tmp_62 = w_19 = w_18 = None
        tmp_64 = torch.nn.functional.dropout(linear_5, 0.0, False, False);  linear_5 = None
        tmp_65 = torch.nn.functional.layer_norm(tmp_64, (64,), w_15, w_14, 1e-05);  tmp_64 = w_15 = w_14 = None
        tmp_66 = tmp_60 + tmp_65;  tmp_60 = tmp_65 = None
        tmp_67 = torch.nn.functional.layer_norm(tmp_66, (64,), w_21, w_20, 1e-05);  tmp_66 = w_21 = w_20 = None
        tmp_68 = tmp_67.transpose(1, 2);  tmp_67 = None
        tmp_69 = torch.adaptive_avg_pool1d(tmp_68, 1);  tmp_68 = None
        tmp_70 = torch.flatten(tmp_69, 1);  tmp_69 = None
        linear_6 = torch.nn.functional.linear(tmp_70, w_1, w_0);  tmp_70 = w_1 = w_0 = None
        return (linear_6,)
        