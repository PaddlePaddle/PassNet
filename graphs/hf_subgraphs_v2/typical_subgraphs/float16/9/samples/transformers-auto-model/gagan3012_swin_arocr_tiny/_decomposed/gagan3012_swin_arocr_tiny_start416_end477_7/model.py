import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, in_0, in_1, in_2, in_3, in_4):
        linear = torch.nn.functional.linear(in_1, w_8, w_7);  in_1 = w_8 = w_7 = None
        tmp_19 = linear.view(64, -1, 12, 32);  linear = None
        tmp_20 = tmp_19.transpose(1, 2);  tmp_19 = None
        tmp_21 = torch.nn.functional.normalize(in_4, dim = -1);  in_4 = None
        tmp_22 = torch.nn.functional.normalize(in_2, dim = -1);  in_2 = None
        tmp_23 = tmp_22.transpose(-2, -1);  tmp_22 = None
        matmul = tmp_21 @ tmp_23;  tmp_21 = tmp_23 = None
        tmp_25 = torch.clamp(w_9, max = 4.605170185988092);  w_9 = None
        tmp_26 = tmp_25.exp();  tmp_25 = None
        tmp_27 = matmul * tmp_26;  matmul = tmp_26 = None
        linear_1 = torch.nn.functional.linear(w_2, w_5, w_4);  w_2 = w_5 = w_4 = None
        tmp_29 = torch.nn.functional.relu(linear_1, inplace = True);  linear_1 = None
        linear_2 = torch.nn.functional.linear(tmp_29, w_6, None);  tmp_29 = w_6 = None
        tmp_31 = linear_2.view(-1, 12);  linear_2 = None
        tmp_32 = w_3.view(-1);  w_3 = None
        tmp_33 = tmp_31[tmp_32];  tmp_31 = tmp_32 = None
        tmp_34 = tmp_33.view(64, 64, -1);  tmp_33 = None
        tmp_35 = tmp_34.permute(2, 0, 1);  tmp_34 = None
        tmp_36 = tmp_35.contiguous();  tmp_35 = None
        tmp_37 = torch.sigmoid(tmp_36);  tmp_36 = None
        tmp_38 = 16 * tmp_37;  tmp_37 = None
        tmp_39 = tmp_38.unsqueeze(0);  tmp_38 = None
        tmp_40 = tmp_27 + tmp_39;  tmp_27 = tmp_39 = None
        tmp_41 = tmp_40.view(1, 64, 12, 64, 64);  tmp_40 = None
        tmp_42 = in_0.unsqueeze(1)
        tmp_43 = tmp_42.unsqueeze(0);  tmp_42 = None
        tmp_44 = tmp_41 + tmp_43;  tmp_41 = tmp_43 = None
        tmp_45 = in_0.unsqueeze(1);  in_0 = None
        tmp_46 = tmp_45.unsqueeze(0);  tmp_45 = None
        tmp_47 = tmp_44 + tmp_46;  tmp_44 = tmp_46 = None
        tmp_48 = tmp_47.view(-1, 12, 64, 64);  tmp_47 = None
        tmp_49 = torch.nn.functional.softmax(tmp_48, dim = -1);  tmp_48 = None
        tmp_50 = torch.nn.functional.dropout(tmp_49, 0.0, False, False);  tmp_49 = None
        matmul_1 = torch.matmul(tmp_50, tmp_20);  tmp_50 = tmp_20 = None
        tmp_52 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_53 = tmp_52.contiguous();  tmp_52 = None
        tmp_54 = tmp_53.view((64, 64, 384));  tmp_53 = None
        linear_3 = torch.nn.functional.linear(tmp_54, w_1, w_0);  tmp_54 = w_1 = w_0 = None
        tmp_56 = torch.nn.functional.dropout(linear_3, 0.0, False, False);  linear_3 = None
        tmp_57 = tmp_56.view(-1, 8, 8, 384);  tmp_56 = None
        tmp_58 = tmp_57.view(-1, 8, 8, 8, 8, 384);  tmp_57 = None
        tmp_59 = tmp_58.permute(0, 1, 3, 2, 4, 5);  tmp_58 = None
        tmp_60 = tmp_59.contiguous();  tmp_59 = None
        tmp_61 = tmp_60.view(-1, 64, 64, 384);  tmp_60 = None
        tmp_62 = torch.roll(tmp_61, shifts = (4, 4), dims = (1, 2));  tmp_61 = None
        tmp_63 = tmp_62.view(1, 4096, 384);  tmp_62 = None
        tmp_64 = torch.nn.functional.layer_norm(tmp_63, (384,), w_15, w_14, 1e-05);  tmp_63 = w_15 = w_14 = None
        tmp_65 = in_3 + tmp_64;  in_3 = tmp_64 = None
        linear_4 = torch.nn.functional.linear(tmp_65, w_11, w_10);  w_11 = w_10 = None
        tmp_67 = torch.nn.functional.gelu(linear_4);  linear_4 = None
        linear_5 = torch.nn.functional.linear(tmp_67, w_17, w_16);  tmp_67 = w_17 = w_16 = None
        tmp_69 = torch.nn.functional.dropout(linear_5, 0.0, False, False);  linear_5 = None
        tmp_70 = torch.nn.functional.layer_norm(tmp_69, (384,), w_13, w_12, 1e-05);  tmp_69 = w_13 = w_12 = None
        tmp_71 = tmp_65 + tmp_70;  tmp_65 = tmp_70 = None
        tmp_72 = tmp_71.view(1, 64, 64, 384)
        tmp_73 = torch.nn.functional.pad(tmp_72, (0, 0, 0, 0, 0, 0), 'constant', None);  tmp_72 = None
        tmp_74 = tmp_73.view(1, 8, 8, 8, 8, 384);  tmp_73 = None
        tmp_75 = tmp_74.permute(0, 1, 3, 2, 4, 5);  tmp_74 = None
        tmp_76 = tmp_75.contiguous();  tmp_75 = None
        tmp_77 = tmp_76.view(-1, 8, 8, 384);  tmp_76 = None
        tmp_78 = tmp_77.view(-1, 64, 384);  tmp_77 = None
        return (tmp_78, tmp_71)
        