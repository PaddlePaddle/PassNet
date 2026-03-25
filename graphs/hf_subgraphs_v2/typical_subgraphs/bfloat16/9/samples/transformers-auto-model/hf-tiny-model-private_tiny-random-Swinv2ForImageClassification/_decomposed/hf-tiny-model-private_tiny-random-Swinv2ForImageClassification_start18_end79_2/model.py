import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, in_0, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_1, w_8, w_7);  in_1 = w_8 = w_7 = None
        tmp_22 = linear.view(64, -1, 2, 8);  linear = None
        tmp_23 = tmp_22.transpose(1, 2);  tmp_22 = None
        tmp_24 = torch.nn.functional.normalize(in_3, dim = -1);  in_3 = None
        tmp_25 = torch.nn.functional.normalize(in_2, dim = -1);  in_2 = None
        tmp_26 = tmp_25.transpose(-2, -1);  tmp_25 = None
        matmul = tmp_24 @ tmp_26;  tmp_24 = tmp_26 = None
        tmp_28 = torch.clamp(w_9, max = 4.605170185988092);  w_9 = None
        tmp_29 = tmp_28.exp();  tmp_28 = None
        tmp_30 = matmul * tmp_29;  matmul = tmp_29 = None
        linear_1 = torch.nn.functional.linear(w_2, w_5, w_4);  w_2 = w_5 = w_4 = None
        tmp_32 = torch.nn.functional.relu(linear_1, inplace = True);  linear_1 = None
        linear_2 = torch.nn.functional.linear(tmp_32, w_6, None);  tmp_32 = w_6 = None
        tmp_34 = linear_2.view(-1, 2);  linear_2 = None
        tmp_35 = w_3.view(-1);  w_3 = None
        tmp_36 = tmp_34[tmp_35];  tmp_34 = tmp_35 = None
        tmp_37 = tmp_36.view(4, 4, -1);  tmp_36 = None
        tmp_38 = tmp_37.permute(2, 0, 1);  tmp_37 = None
        tmp_39 = tmp_38.contiguous();  tmp_38 = None
        tmp_40 = torch.sigmoid(tmp_39);  tmp_39 = None
        tmp_41 = 16 * tmp_40;  tmp_40 = None
        tmp_42 = tmp_41.unsqueeze(0);  tmp_41 = None
        tmp_43 = tmp_30 + tmp_42;  tmp_30 = tmp_42 = None
        tmp_44 = torch.nn.functional.softmax(tmp_43, dim = -1);  tmp_43 = None
        tmp_45 = torch.nn.functional.dropout(tmp_44, 0.0, False, False);  tmp_44 = None
        matmul_1 = torch.matmul(tmp_45, tmp_23);  tmp_45 = tmp_23 = None
        tmp_47 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_48 = tmp_47.contiguous();  tmp_47 = None
        tmp_49 = tmp_48.view((64, 4, 16));  tmp_48 = None
        linear_3 = torch.nn.functional.linear(tmp_49, w_1, w_0);  tmp_49 = w_1 = w_0 = None
        tmp_51 = torch.nn.functional.dropout(linear_3, 0.0, False, False);  linear_3 = None
        tmp_52 = tmp_51.view(-1, 2, 2, 16);  tmp_51 = None
        tmp_53 = tmp_52.view(-1, 8, 8, 2, 2, 16);  tmp_52 = None
        tmp_54 = tmp_53.permute(0, 1, 3, 2, 4, 5);  tmp_53 = None
        tmp_55 = tmp_54.contiguous();  tmp_54 = None
        tmp_56 = tmp_55.view(-1, 16, 16, 16);  tmp_55 = None
        tmp_57 = tmp_56.view(1, 256, 16);  tmp_56 = None
        tmp_58 = torch.nn.functional.layer_norm(tmp_57, (16,), w_15, w_14, 1e-05);  tmp_57 = w_15 = w_14 = None
        tmp_59 = in_0 + tmp_58;  in_0 = tmp_58 = None
        linear_4 = torch.nn.functional.linear(tmp_59, w_11, w_10);  w_11 = w_10 = None
        tmp_61 = torch.nn.functional.gelu(linear_4);  linear_4 = None
        linear_5 = torch.nn.functional.linear(tmp_61, w_17, w_16);  tmp_61 = w_17 = w_16 = None
        tmp_63 = torch.nn.functional.dropout(linear_5, 0.0, False, False);  linear_5 = None
        tmp_64 = torch.nn.functional.layer_norm(tmp_63, (16,), w_13, w_12, 1e-05);  tmp_63 = w_13 = w_12 = None
        tmp_65 = tmp_59 + tmp_64;  tmp_59 = tmp_64 = None
        tmp_66 = tmp_65.view(1, 16, 16, 16);  tmp_65 = None
        tmp_67 = tmp_66[(slice(None, None, None), slice(0, None, 2), slice(0, None, 2), slice(None, None, None))]
        tmp_68 = tmp_66[(slice(None, None, None), slice(1, None, 2), slice(0, None, 2), slice(None, None, None))]
        tmp_69 = tmp_66[(slice(None, None, None), slice(0, None, 2), slice(1, None, 2), slice(None, None, None))]
        tmp_70 = tmp_66[(slice(None, None, None), slice(1, None, 2), slice(1, None, 2), slice(None, None, None))];  tmp_66 = None
        tmp_71 = torch.cat([tmp_67, tmp_68, tmp_69, tmp_70], -1);  tmp_67 = tmp_68 = tmp_69 = tmp_70 = None
        tmp_72 = tmp_71.view(1, -1, 64);  tmp_71 = None
        linear_6 = torch.nn.functional.linear(tmp_72, w_20, None);  tmp_72 = w_20 = None
        tmp_74 = torch.nn.functional.layer_norm(linear_6, (32,), w_19, w_18, 1e-05);  linear_6 = w_19 = w_18 = None
        tmp_75 = tmp_74.view(1, 8, 8, 32)
        tmp_76 = torch.nn.functional.pad(tmp_75, (0, 0, 0, 0, 0, 0), 'constant', None);  tmp_75 = None
        tmp_77 = tmp_76.view(1, 4, 2, 4, 2, 32);  tmp_76 = None
        tmp_78 = tmp_77.permute(0, 1, 3, 2, 4, 5);  tmp_77 = None
        tmp_79 = tmp_78.contiguous();  tmp_78 = None
        tmp_80 = tmp_79.view(-1, 2, 2, 32);  tmp_79 = None
        tmp_81 = tmp_80.view(-1, 4, 32);  tmp_80 = None
        return (tmp_81, tmp_74)
        