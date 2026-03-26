import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, w_28 : torch.Tensor, w_29 : torch.Tensor, w_30 : torch.Tensor, w_31 : torch.Tensor, w_32 : torch.Tensor, w_33 : torch.Tensor, w_34 : torch.Tensor, w_35 : torch.Tensor, w_36 : torch.Tensor):
        tmp_38 = in_0[(slice(None, None, None), None)];  in_0 = None
        conv1d = torch.conv1d(tmp_38, w_6, w_5, (5,), (0,), (1,), 1);  tmp_38 = w_6 = w_5 = None
        tmp_40 = conv1d.transpose(-2, -1);  conv1d = None
        tmp_41 = torch.nn.functional.layer_norm(tmp_40, (512,), w_8, w_7, 1e-05);  tmp_40 = w_8 = w_7 = None
        tmp_42 = tmp_41.transpose(-2, -1);  tmp_41 = None
        tmp_43 = torch.nn.functional.gelu(tmp_42);  tmp_42 = None
        conv1d_1 = torch.conv1d(tmp_43, w_10, w_9, (2,), (0,), (1,), 1);  tmp_43 = w_10 = w_9 = None
        tmp_45 = conv1d_1.transpose(-2, -1);  conv1d_1 = None
        tmp_46 = torch.nn.functional.layer_norm(tmp_45, (512,), w_12, w_11, 1e-05);  tmp_45 = w_12 = w_11 = None
        tmp_47 = tmp_46.transpose(-2, -1);  tmp_46 = None
        tmp_48 = torch.nn.functional.gelu(tmp_47);  tmp_47 = None
        conv1d_2 = torch.conv1d(tmp_48, w_14, w_13, (2,), (0,), (1,), 1);  tmp_48 = w_14 = w_13 = None
        tmp_50 = conv1d_2.transpose(-2, -1);  conv1d_2 = None
        tmp_51 = torch.nn.functional.layer_norm(tmp_50, (512,), w_16, w_15, 1e-05);  tmp_50 = w_16 = w_15 = None
        tmp_52 = tmp_51.transpose(-2, -1);  tmp_51 = None
        tmp_53 = torch.nn.functional.gelu(tmp_52);  tmp_52 = None
        conv1d_3 = torch.conv1d(tmp_53, w_18, w_17, (2,), (0,), (1,), 1);  tmp_53 = w_18 = w_17 = None
        tmp_55 = conv1d_3.transpose(-2, -1);  conv1d_3 = None
        tmp_56 = torch.nn.functional.layer_norm(tmp_55, (512,), w_20, w_19, 1e-05);  tmp_55 = w_20 = w_19 = None
        tmp_57 = tmp_56.transpose(-2, -1);  tmp_56 = None
        tmp_58 = torch.nn.functional.gelu(tmp_57);  tmp_57 = None
        conv1d_4 = torch.conv1d(tmp_58, w_22, w_21, (2,), (0,), (1,), 1);  tmp_58 = w_22 = w_21 = None
        tmp_60 = conv1d_4.transpose(-2, -1);  conv1d_4 = None
        tmp_61 = torch.nn.functional.layer_norm(tmp_60, (512,), w_24, w_23, 1e-05);  tmp_60 = w_24 = w_23 = None
        tmp_62 = tmp_61.transpose(-2, -1);  tmp_61 = None
        tmp_63 = torch.nn.functional.gelu(tmp_62);  tmp_62 = None
        conv1d_5 = torch.conv1d(tmp_63, w_26, w_25, (2,), (0,), (1,), 1);  tmp_63 = w_26 = w_25 = None
        tmp_65 = conv1d_5.transpose(-2, -1);  conv1d_5 = None
        tmp_66 = torch.nn.functional.layer_norm(tmp_65, (512,), w_28, w_27, 1e-05);  tmp_65 = w_28 = w_27 = None
        tmp_67 = tmp_66.transpose(-2, -1);  tmp_66 = None
        tmp_68 = torch.nn.functional.gelu(tmp_67);  tmp_67 = None
        conv1d_6 = torch.conv1d(tmp_68, w_30, w_29, (2,), (0,), (1,), 1);  tmp_68 = w_30 = w_29 = None
        tmp_70 = conv1d_6.transpose(-2, -1);  conv1d_6 = None
        tmp_71 = torch.nn.functional.layer_norm(tmp_70, (512,), w_32, w_31, 1e-05);  tmp_70 = w_32 = w_31 = None
        tmp_72 = tmp_71.transpose(-2, -1);  tmp_71 = None
        tmp_73 = torch.nn.functional.gelu(tmp_72);  tmp_72 = None
        tmp_74 = tmp_73.transpose(1, 2);  tmp_73 = None
        tmp_75 = torch.nn.functional.layer_norm(tmp_74, (512,), w_34, w_33, 1e-05);  tmp_74 = w_34 = w_33 = None
        linear = torch.nn.functional.linear(tmp_75, w_36, w_35);  tmp_75 = w_36 = w_35 = None
        tmp_77 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_78 = tmp_77.transpose(1, 2)
        tmp_79 = torch._weight_norm(w_3, w_2, 2);  w_3 = w_2 = None
        conv1d_7 = torch.conv1d(tmp_78, tmp_79, w_4, (1,), (64,), (1,), 16);  tmp_78 = tmp_79 = w_4 = None
        tmp_81 = conv1d_7[(slice(None, None, None), slice(None, None, None), slice(None, -1, None))];  conv1d_7 = None
        tmp_82 = torch.nn.functional.gelu(tmp_81);  tmp_81 = None
        tmp_83 = tmp_82.transpose(1, 2);  tmp_82 = None
        tmp_84 = tmp_77 + tmp_83;  tmp_77 = tmp_83 = None
        tmp_85 = torch.nn.functional.dropout(tmp_84, 0.1, False, False);  tmp_84 = None
        tmp_86 = torch.rand([]);  tmp_86 = None
        tmp_87 = torch.nn.functional.layer_norm(tmp_85, (1024,), w_1, w_0, 1e-05);  w_1 = w_0 = None
        return (tmp_85, tmp_87)
        