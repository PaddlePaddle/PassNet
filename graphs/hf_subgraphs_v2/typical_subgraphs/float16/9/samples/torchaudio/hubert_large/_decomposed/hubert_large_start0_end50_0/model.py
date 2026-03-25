import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, w_28 : torch.Tensor, w_29 : torch.Tensor, in_0 : torch.Tensor):
        tmp_31 = torch.nn.functional.layer_norm(in_0, (1, 80000));  in_0 = None
        tmp_32 = tmp_31.unsqueeze(1);  tmp_31 = None
        conv1d = torch.conv1d(tmp_32, w_9, None, (5,), (0,), (1,), 1);  tmp_32 = w_9 = None
        tmp_34 = conv1d.transpose(-2, -1);  conv1d = None
        tmp_35 = torch.nn.functional.layer_norm(tmp_34, (512,), w_11, w_10, 1e-05);  tmp_34 = w_11 = w_10 = None
        tmp_36 = tmp_35.transpose(-2, -1);  tmp_35 = None
        tmp_37 = torch.nn.functional.gelu(tmp_36);  tmp_36 = None
        conv1d_1 = torch.conv1d(tmp_37, w_12, None, (2,), (0,), (1,), 1);  tmp_37 = w_12 = None
        tmp_39 = conv1d_1.transpose(-2, -1);  conv1d_1 = None
        tmp_40 = torch.nn.functional.layer_norm(tmp_39, (512,), w_14, w_13, 1e-05);  tmp_39 = w_14 = w_13 = None
        tmp_41 = tmp_40.transpose(-2, -1);  tmp_40 = None
        tmp_42 = torch.nn.functional.gelu(tmp_41);  tmp_41 = None
        conv1d_2 = torch.conv1d(tmp_42, w_15, None, (2,), (0,), (1,), 1);  tmp_42 = w_15 = None
        tmp_44 = conv1d_2.transpose(-2, -1);  conv1d_2 = None
        tmp_45 = torch.nn.functional.layer_norm(tmp_44, (512,), w_17, w_16, 1e-05);  tmp_44 = w_17 = w_16 = None
        tmp_46 = tmp_45.transpose(-2, -1);  tmp_45 = None
        tmp_47 = torch.nn.functional.gelu(tmp_46);  tmp_46 = None
        conv1d_3 = torch.conv1d(tmp_47, w_18, None, (2,), (0,), (1,), 1);  tmp_47 = w_18 = None
        tmp_49 = conv1d_3.transpose(-2, -1);  conv1d_3 = None
        tmp_50 = torch.nn.functional.layer_norm(tmp_49, (512,), w_20, w_19, 1e-05);  tmp_49 = w_20 = w_19 = None
        tmp_51 = tmp_50.transpose(-2, -1);  tmp_50 = None
        tmp_52 = torch.nn.functional.gelu(tmp_51);  tmp_51 = None
        conv1d_4 = torch.conv1d(tmp_52, w_21, None, (2,), (0,), (1,), 1);  tmp_52 = w_21 = None
        tmp_54 = conv1d_4.transpose(-2, -1);  conv1d_4 = None
        tmp_55 = torch.nn.functional.layer_norm(tmp_54, (512,), w_23, w_22, 1e-05);  tmp_54 = w_23 = w_22 = None
        tmp_56 = tmp_55.transpose(-2, -1);  tmp_55 = None
        tmp_57 = torch.nn.functional.gelu(tmp_56);  tmp_56 = None
        conv1d_5 = torch.conv1d(tmp_57, w_24, None, (2,), (0,), (1,), 1);  tmp_57 = w_24 = None
        tmp_59 = conv1d_5.transpose(-2, -1);  conv1d_5 = None
        tmp_60 = torch.nn.functional.layer_norm(tmp_59, (512,), w_26, w_25, 1e-05);  tmp_59 = w_26 = w_25 = None
        tmp_61 = tmp_60.transpose(-2, -1);  tmp_60 = None
        tmp_62 = torch.nn.functional.gelu(tmp_61);  tmp_61 = None
        conv1d_6 = torch.conv1d(tmp_62, w_27, None, (2,), (0,), (1,), 1);  tmp_62 = w_27 = None
        tmp_64 = conv1d_6.transpose(-2, -1);  conv1d_6 = None
        tmp_65 = torch.nn.functional.layer_norm(tmp_64, (512,), w_29, w_28, 1e-05);  tmp_64 = w_29 = w_28 = None
        tmp_66 = tmp_65.transpose(-2, -1);  tmp_65 = None
        tmp_67 = torch.nn.functional.gelu(tmp_66);  tmp_66 = None
        tmp_68 = tmp_67.transpose(1, 2);  tmp_67 = None
        tmp_69 = torch.nn.functional.layer_norm(tmp_68, (512,), w_1, w_0, 1e-05);  tmp_68 = w_1 = w_0 = None
        linear = torch.nn.functional.linear(tmp_69, w_3, w_2);  tmp_69 = w_3 = w_2 = None
        tmp_71 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_72 = tmp_71.transpose(-2, -1)
        tmp_73 = torch._weight_norm(w_7, w_6, 2);  w_7 = w_6 = None
        conv1d_7 = torch.conv1d(tmp_72, tmp_73, w_8, (1,), (64,), (1,), 16);  tmp_72 = tmp_73 = w_8 = None
        tmp_75 = conv1d_7[(Ellipsis, slice(None, -1, None))];  conv1d_7 = None
        tmp_76 = torch.nn.functional.gelu(tmp_75);  tmp_75 = None
        tmp_77 = tmp_76.transpose(-2, -1);  tmp_76 = None
        tmp_78 = tmp_71 + tmp_77;  tmp_71 = tmp_77 = None
        tmp_79 = torch.nn.functional.dropout(tmp_78, 0.0, False, False);  tmp_78 = None
        tmp_80 = torch.nn.functional.layer_norm(tmp_79, (1024,), w_5, w_4, 1e-05);  w_5 = w_4 = None
        return (tmp_79, tmp_80)
        