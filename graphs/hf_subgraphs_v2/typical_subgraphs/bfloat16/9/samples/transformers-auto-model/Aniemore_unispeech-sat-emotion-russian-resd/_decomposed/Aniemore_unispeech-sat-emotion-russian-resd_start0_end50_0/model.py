import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, w_28 : torch.Tensor, w_29 : torch.Tensor):
        tmp_31 = in_0[(slice(None, None, None), None)];  in_0 = None
        conv1d = torch.conv1d(tmp_31, w_5, None, (5,), (0,), (1,), 1);  tmp_31 = w_5 = None
        tmp_33 = conv1d.transpose(-2, -1);  conv1d = None
        tmp_34 = torch.nn.functional.layer_norm(tmp_33, (512,), w_7, w_6, 1e-05);  tmp_33 = w_7 = w_6 = None
        tmp_35 = tmp_34.transpose(-2, -1);  tmp_34 = None
        tmp_36 = torch.nn.functional.gelu(tmp_35);  tmp_35 = None
        conv1d_1 = torch.conv1d(tmp_36, w_8, None, (2,), (0,), (1,), 1);  tmp_36 = w_8 = None
        tmp_38 = conv1d_1.transpose(-2, -1);  conv1d_1 = None
        tmp_39 = torch.nn.functional.layer_norm(tmp_38, (512,), w_10, w_9, 1e-05);  tmp_38 = w_10 = w_9 = None
        tmp_40 = tmp_39.transpose(-2, -1);  tmp_39 = None
        tmp_41 = torch.nn.functional.gelu(tmp_40);  tmp_40 = None
        conv1d_2 = torch.conv1d(tmp_41, w_11, None, (2,), (0,), (1,), 1);  tmp_41 = w_11 = None
        tmp_43 = conv1d_2.transpose(-2, -1);  conv1d_2 = None
        tmp_44 = torch.nn.functional.layer_norm(tmp_43, (512,), w_13, w_12, 1e-05);  tmp_43 = w_13 = w_12 = None
        tmp_45 = tmp_44.transpose(-2, -1);  tmp_44 = None
        tmp_46 = torch.nn.functional.gelu(tmp_45);  tmp_45 = None
        conv1d_3 = torch.conv1d(tmp_46, w_14, None, (2,), (0,), (1,), 1);  tmp_46 = w_14 = None
        tmp_48 = conv1d_3.transpose(-2, -1);  conv1d_3 = None
        tmp_49 = torch.nn.functional.layer_norm(tmp_48, (512,), w_16, w_15, 1e-05);  tmp_48 = w_16 = w_15 = None
        tmp_50 = tmp_49.transpose(-2, -1);  tmp_49 = None
        tmp_51 = torch.nn.functional.gelu(tmp_50);  tmp_50 = None
        conv1d_4 = torch.conv1d(tmp_51, w_17, None, (2,), (0,), (1,), 1);  tmp_51 = w_17 = None
        tmp_53 = conv1d_4.transpose(-2, -1);  conv1d_4 = None
        tmp_54 = torch.nn.functional.layer_norm(tmp_53, (512,), w_19, w_18, 1e-05);  tmp_53 = w_19 = w_18 = None
        tmp_55 = tmp_54.transpose(-2, -1);  tmp_54 = None
        tmp_56 = torch.nn.functional.gelu(tmp_55);  tmp_55 = None
        conv1d_5 = torch.conv1d(tmp_56, w_20, None, (2,), (0,), (1,), 1);  tmp_56 = w_20 = None
        tmp_58 = conv1d_5.transpose(-2, -1);  conv1d_5 = None
        tmp_59 = torch.nn.functional.layer_norm(tmp_58, (512,), w_22, w_21, 1e-05);  tmp_58 = w_22 = w_21 = None
        tmp_60 = tmp_59.transpose(-2, -1);  tmp_59 = None
        tmp_61 = torch.nn.functional.gelu(tmp_60);  tmp_60 = None
        conv1d_6 = torch.conv1d(tmp_61, w_23, None, (2,), (0,), (1,), 1);  tmp_61 = w_23 = None
        tmp_63 = conv1d_6.transpose(-2, -1);  conv1d_6 = None
        tmp_64 = torch.nn.functional.layer_norm(tmp_63, (512,), w_25, w_24, 1e-05);  tmp_63 = w_25 = w_24 = None
        tmp_65 = tmp_64.transpose(-2, -1);  tmp_64 = None
        tmp_66 = torch.nn.functional.gelu(tmp_65);  tmp_65 = None
        tmp_67 = tmp_66.transpose(1, 2);  tmp_66 = None
        tmp_68 = torch.nn.functional.layer_norm(tmp_67, (512,), w_27, w_26, 1e-05);  tmp_67 = w_27 = w_26 = None
        linear = torch.nn.functional.linear(tmp_68, w_29, w_28);  tmp_68 = w_29 = w_28 = None
        tmp_70 = torch.nn.functional.dropout(linear, 0.05, False, False);  linear = None
        tmp_71 = tmp_70.transpose(1, 2)
        tmp_72 = torch._weight_norm(w_3, w_2, 2);  w_3 = w_2 = None
        conv1d_7 = torch.conv1d(tmp_71, tmp_72, w_4, (1,), (64,), (1,), 16);  tmp_71 = tmp_72 = w_4 = None
        tmp_74 = conv1d_7[(slice(None, None, None), slice(None, None, None), slice(None, -1, None))];  conv1d_7 = None
        tmp_75 = torch.nn.functional.gelu(tmp_74);  tmp_74 = None
        tmp_76 = tmp_75.transpose(1, 2);  tmp_75 = None
        tmp_77 = tmp_70 + tmp_76;  tmp_70 = tmp_76 = None
        tmp_78 = torch.nn.functional.dropout(tmp_77, 0.05, False, False);  tmp_77 = None
        tmp_79 = torch.rand([]);  tmp_79 = None
        tmp_80 = torch.nn.functional.layer_norm(tmp_78, (1024,), w_1, w_0, 1e-05);  w_1 = w_0 = None
        return (tmp_78, tmp_80)
        