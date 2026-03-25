import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, w_28 : torch.Tensor, w_29 : torch.Tensor, w_30 : torch.Tensor, w_31 : torch.Tensor, w_32 : torch.Tensor, w_33 : torch.Tensor, w_34 : torch.Tensor, w_35 : torch.Tensor, w_36 : torch.Tensor, in_0 : torch.Tensor):
        tmp_38 = torch.nn.functional.layer_norm(in_0, (1, 80000));  in_0 = None
        tmp_39 = tmp_38.unsqueeze(1);  tmp_38 = None
        conv1d = torch.conv1d(tmp_39, w_10, w_9, (5,), (0,), (1,), 1);  tmp_39 = w_10 = w_9 = None
        tmp_41 = conv1d.transpose(-2, -1);  conv1d = None
        tmp_42 = torch.nn.functional.layer_norm(tmp_41, (512,), w_12, w_11, 1e-05);  tmp_41 = w_12 = w_11 = None
        tmp_43 = tmp_42.transpose(-2, -1);  tmp_42 = None
        tmp_44 = torch.nn.functional.gelu(tmp_43);  tmp_43 = None
        conv1d_1 = torch.conv1d(tmp_44, w_14, w_13, (2,), (0,), (1,), 1);  tmp_44 = w_14 = w_13 = None
        tmp_46 = conv1d_1.transpose(-2, -1);  conv1d_1 = None
        tmp_47 = torch.nn.functional.layer_norm(tmp_46, (512,), w_16, w_15, 1e-05);  tmp_46 = w_16 = w_15 = None
        tmp_48 = tmp_47.transpose(-2, -1);  tmp_47 = None
        tmp_49 = torch.nn.functional.gelu(tmp_48);  tmp_48 = None
        conv1d_2 = torch.conv1d(tmp_49, w_18, w_17, (2,), (0,), (1,), 1);  tmp_49 = w_18 = w_17 = None
        tmp_51 = conv1d_2.transpose(-2, -1);  conv1d_2 = None
        tmp_52 = torch.nn.functional.layer_norm(tmp_51, (512,), w_20, w_19, 1e-05);  tmp_51 = w_20 = w_19 = None
        tmp_53 = tmp_52.transpose(-2, -1);  tmp_52 = None
        tmp_54 = torch.nn.functional.gelu(tmp_53);  tmp_53 = None
        conv1d_3 = torch.conv1d(tmp_54, w_22, w_21, (2,), (0,), (1,), 1);  tmp_54 = w_22 = w_21 = None
        tmp_56 = conv1d_3.transpose(-2, -1);  conv1d_3 = None
        tmp_57 = torch.nn.functional.layer_norm(tmp_56, (512,), w_24, w_23, 1e-05);  tmp_56 = w_24 = w_23 = None
        tmp_58 = tmp_57.transpose(-2, -1);  tmp_57 = None
        tmp_59 = torch.nn.functional.gelu(tmp_58);  tmp_58 = None
        conv1d_4 = torch.conv1d(tmp_59, w_26, w_25, (2,), (0,), (1,), 1);  tmp_59 = w_26 = w_25 = None
        tmp_61 = conv1d_4.transpose(-2, -1);  conv1d_4 = None
        tmp_62 = torch.nn.functional.layer_norm(tmp_61, (512,), w_28, w_27, 1e-05);  tmp_61 = w_28 = w_27 = None
        tmp_63 = tmp_62.transpose(-2, -1);  tmp_62 = None
        tmp_64 = torch.nn.functional.gelu(tmp_63);  tmp_63 = None
        conv1d_5 = torch.conv1d(tmp_64, w_30, w_29, (2,), (0,), (1,), 1);  tmp_64 = w_30 = w_29 = None
        tmp_66 = conv1d_5.transpose(-2, -1);  conv1d_5 = None
        tmp_67 = torch.nn.functional.layer_norm(tmp_66, (512,), w_32, w_31, 1e-05);  tmp_66 = w_32 = w_31 = None
        tmp_68 = tmp_67.transpose(-2, -1);  tmp_67 = None
        tmp_69 = torch.nn.functional.gelu(tmp_68);  tmp_68 = None
        conv1d_6 = torch.conv1d(tmp_69, w_34, w_33, (2,), (0,), (1,), 1);  tmp_69 = w_34 = w_33 = None
        tmp_71 = conv1d_6.transpose(-2, -1);  conv1d_6 = None
        tmp_72 = torch.nn.functional.layer_norm(tmp_71, (512,), w_36, w_35, 1e-05);  tmp_71 = w_36 = w_35 = None
        tmp_73 = tmp_72.transpose(-2, -1);  tmp_72 = None
        tmp_74 = torch.nn.functional.gelu(tmp_73);  tmp_73 = None
        tmp_75 = tmp_74.transpose(1, 2);  tmp_74 = None
        tmp_76 = torch.nn.functional.layer_norm(tmp_75, (512,), w_1, w_0, 1e-05);  tmp_75 = w_1 = w_0 = None
        linear = torch.nn.functional.linear(tmp_76, w_3, w_2);  tmp_76 = w_3 = w_2 = None
        tmp_78 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_79 = tmp_78.transpose(-2, -1)
        tmp_80 = torch._weight_norm(w_7, w_6, 2);  w_7 = w_6 = None
        conv1d_7 = torch.conv1d(tmp_79, tmp_80, w_8, (1,), (64,), (1,), 16);  tmp_79 = tmp_80 = w_8 = None
        tmp_82 = conv1d_7[(Ellipsis, slice(None, -1, None))];  conv1d_7 = None
        tmp_83 = torch.nn.functional.gelu(tmp_82);  tmp_82 = None
        tmp_84 = tmp_83.transpose(-2, -1);  tmp_83 = None
        tmp_85 = tmp_78 + tmp_84;  tmp_78 = tmp_84 = None
        tmp_86 = torch.nn.functional.dropout(tmp_85, 0.0, False, False);  tmp_85 = None
        tmp_87 = torch.nn.functional.layer_norm(tmp_86, (1024,), w_5, w_4, 1e-05);  w_5 = w_4 = None
        return (tmp_86, tmp_87)
        