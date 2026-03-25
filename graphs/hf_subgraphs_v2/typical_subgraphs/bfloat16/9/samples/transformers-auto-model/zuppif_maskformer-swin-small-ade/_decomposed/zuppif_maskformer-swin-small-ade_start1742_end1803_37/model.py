import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, in_0, in_1, in_2, in_3, in_4):
        tmp_20 = torch.nn.functional.relu(in_2, inplace = False);  in_2 = None
        tmp_21 = torch.nn.functional.dropout(tmp_20, p = 0.0, training = False);  tmp_20 = None
        linear = torch.nn.functional.linear(tmp_21, w_1, w_0);  tmp_21 = w_1 = w_0 = None
        tmp_23 = torch.nn.functional.dropout(linear, p = 0.1, training = False);  linear = None
        tmp_24 = in_0 + tmp_23;  in_0 = tmp_23 = None
        tmp_25 = torch.nn.functional.layer_norm(tmp_24, (256,), w_3, w_2, 1e-05);  tmp_24 = w_3 = w_2 = None
        tmp_26 = tmp_25 + in_4
        linear_1 = torch.nn.functional.linear(tmp_26, w_17, w_16);  w_17 = w_16 = None
        tmp_28 = linear_1 * 0.1767766952966369;  linear_1 = None
        linear_2 = torch.nn.functional.linear(tmp_26, w_13, w_12);  tmp_26 = w_13 = w_12 = None
        tmp_30 = linear_2.view(1, -1, 8, 32);  linear_2 = None
        tmp_31 = tmp_30.transpose(1, 2);  tmp_30 = None
        tmp_32 = tmp_31.contiguous();  tmp_31 = None
        linear_3 = torch.nn.functional.linear(tmp_25, w_19, w_18);  w_19 = w_18 = None
        tmp_34 = linear_3.view(1, -1, 8, 32);  linear_3 = None
        tmp_35 = tmp_34.transpose(1, 2);  tmp_34 = None
        tmp_36 = tmp_35.contiguous();  tmp_35 = None
        tmp_37 = tmp_28.view(1, 100, 8, 32);  tmp_28 = None
        tmp_38 = tmp_37.transpose(1, 2);  tmp_37 = None
        tmp_39 = tmp_38.contiguous();  tmp_38 = None
        tmp_40 = tmp_39.view(8, -1, 32);  tmp_39 = None
        tmp_41 = tmp_32.view(8, -1, 32);  tmp_32 = None
        tmp_42 = tmp_36.view(8, -1, 32);  tmp_36 = None
        tmp_43 = tmp_41.transpose(1, 2);  tmp_41 = None
        bmm = torch.bmm(tmp_40, tmp_43);  tmp_40 = tmp_43 = None
        tmp_45 = torch.nn.functional.softmax(bmm, dim = -1);  bmm = None
        tmp_46 = torch.nn.functional.dropout(tmp_45, p = 0.0, training = False);  tmp_45 = None
        bmm_1 = torch.bmm(tmp_46, tmp_42);  tmp_46 = tmp_42 = None
        tmp_48 = bmm_1.view(1, 8, 100, 32);  bmm_1 = None
        tmp_49 = tmp_48.transpose(1, 2);  tmp_48 = None
        tmp_50 = tmp_49.reshape(1, 100, 256);  tmp_49 = None
        linear_4 = torch.nn.functional.linear(tmp_50, w_15, w_14);  tmp_50 = w_15 = w_14 = None
        tmp_52 = torch.nn.functional.dropout(linear_4, p = 0.1, training = False);  linear_4 = None
        tmp_53 = tmp_25 + tmp_52;  tmp_25 = tmp_52 = None
        tmp_54 = torch.nn.functional.layer_norm(tmp_53, (256,), w_11, w_10, 1e-05);  tmp_53 = w_11 = w_10 = None
        tmp_55 = tmp_54 + in_4;  in_4 = None
        tmp_56 = in_1 + in_3;  in_3 = None
        linear_5 = torch.nn.functional.linear(tmp_55, w_7, w_6);  tmp_55 = w_7 = w_6 = None
        tmp_58 = linear_5 * 0.1767766952966369;  linear_5 = None
        linear_6 = torch.nn.functional.linear(tmp_56, w_5, w_4);  tmp_56 = w_5 = w_4 = None
        tmp_60 = linear_6.view(1, -1, 8, 32);  linear_6 = None
        tmp_61 = tmp_60.transpose(1, 2);  tmp_60 = None
        tmp_62 = tmp_61.contiguous();  tmp_61 = None
        linear_7 = torch.nn.functional.linear(in_1, w_9, w_8);  in_1 = w_9 = w_8 = None
        tmp_64 = linear_7.view(1, -1, 8, 32);  linear_7 = None
        tmp_65 = tmp_64.transpose(1, 2);  tmp_64 = None
        tmp_66 = tmp_65.contiguous();  tmp_65 = None
        tmp_67 = tmp_58.view(1, 100, 8, 32);  tmp_58 = None
        tmp_68 = tmp_67.transpose(1, 2);  tmp_67 = None
        tmp_69 = tmp_68.contiguous();  tmp_68 = None
        tmp_70 = tmp_69.view(8, -1, 32);  tmp_69 = None
        tmp_71 = tmp_62.view(8, -1, 32);  tmp_62 = None
        tmp_72 = tmp_66.view(8, -1, 32);  tmp_66 = None
        tmp_73 = tmp_71.transpose(1, 2);  tmp_71 = None
        bmm_2 = torch.bmm(tmp_70, tmp_73);  tmp_70 = tmp_73 = None
        tmp_75 = torch.nn.functional.softmax(bmm_2, dim = -1);  bmm_2 = None
        tmp_76 = torch.nn.functional.dropout(tmp_75, p = 0.0, training = False);  tmp_75 = None
        bmm_3 = torch.bmm(tmp_76, tmp_72);  tmp_76 = tmp_72 = None
        tmp_78 = bmm_3.view(1, 8, 100, 32);  bmm_3 = None
        tmp_79 = tmp_78.transpose(1, 2);  tmp_78 = None
        tmp_80 = tmp_79.reshape(1, 100, 256);  tmp_79 = None
        return (tmp_80, tmp_54)
        