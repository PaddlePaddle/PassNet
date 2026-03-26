import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor):
        tmp_25 = in_0[(slice(None, None, None), None)];  in_0 = None
        conv1d = torch.conv1d(tmp_25, w_5, None, (5,), (0,), (1,), 1);  tmp_25 = w_5 = None
        tmp_27 = torch.nn.functional.group_norm(conv1d, 64, w_7, w_6, 1e-05);  conv1d = w_7 = w_6 = None
        tmp_28 = torch.nn.functional.gelu(tmp_27);  tmp_27 = None
        conv1d_1 = torch.conv1d(tmp_28, w_11, None, (2,), (0,), (1,), 1);  tmp_28 = w_11 = None
        tmp_30 = torch.nn.functional.gelu(conv1d_1);  conv1d_1 = None
        conv1d_2 = torch.conv1d(tmp_30, w_12, None, (1,), (0,), (1,), 1);  tmp_30 = w_12 = None
        tmp_32 = torch.nn.functional.gelu(conv1d_2);  conv1d_2 = None
        conv1d_3 = torch.conv1d(tmp_32, w_13, None, (2,), (0,), (1,), 1);  tmp_32 = w_13 = None
        tmp_34 = torch.nn.functional.gelu(conv1d_3);  conv1d_3 = None
        conv1d_4 = torch.conv1d(tmp_34, w_14, None, (1,), (0,), (1,), 1);  tmp_34 = w_14 = None
        tmp_36 = torch.nn.functional.gelu(conv1d_4);  conv1d_4 = None
        conv1d_5 = torch.conv1d(tmp_36, w_15, None, (2,), (0,), (1,), 1);  tmp_36 = w_15 = None
        tmp_38 = torch.nn.functional.gelu(conv1d_5);  conv1d_5 = None
        conv1d_6 = torch.conv1d(tmp_38, w_16, None, (1,), (0,), (1,), 1);  tmp_38 = w_16 = None
        tmp_40 = torch.nn.functional.gelu(conv1d_6);  conv1d_6 = None
        conv1d_7 = torch.conv1d(tmp_40, w_17, None, (2,), (0,), (1,), 1);  tmp_40 = w_17 = None
        tmp_42 = torch.nn.functional.gelu(conv1d_7);  conv1d_7 = None
        conv1d_8 = torch.conv1d(tmp_42, w_18, None, (1,), (0,), (1,), 1);  tmp_42 = w_18 = None
        tmp_44 = torch.nn.functional.gelu(conv1d_8);  conv1d_8 = None
        conv1d_9 = torch.conv1d(tmp_44, w_19, None, (2,), (0,), (1,), 1);  tmp_44 = w_19 = None
        tmp_46 = torch.nn.functional.gelu(conv1d_9);  conv1d_9 = None
        conv1d_10 = torch.conv1d(tmp_46, w_8, None, (1,), (0,), (1,), 1);  tmp_46 = w_8 = None
        tmp_48 = torch.nn.functional.gelu(conv1d_10);  conv1d_10 = None
        conv1d_11 = torch.conv1d(tmp_48, w_9, None, (2,), (0,), (1,), 1);  tmp_48 = w_9 = None
        tmp_50 = torch.nn.functional.gelu(conv1d_11);  conv1d_11 = None
        conv1d_12 = torch.conv1d(tmp_50, w_10, None, (1,), (0,), (1,), 1);  tmp_50 = w_10 = None
        tmp_52 = torch.nn.functional.gelu(conv1d_12);  conv1d_12 = None
        tmp_53 = tmp_52.transpose(1, 2);  tmp_52 = None
        tmp_54 = torch.nn.functional.layer_norm(tmp_53, (512,), w_23, w_22, 1e-05);  tmp_53 = w_23 = w_22 = None
        linear = torch.nn.functional.linear(tmp_54, w_21, w_20);  tmp_54 = w_21 = w_20 = None
        tmp_56 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_57 = tmp_56.transpose(1, 2);  tmp_56 = None
        tmp_58 = torch._weight_norm(w_3, w_2, 2);  w_3 = w_2 = None
        conv1d_13 = torch.conv1d(tmp_57, tmp_58, w_4, (2,), (15,), (1,), 16);  tmp_58 = w_4 = None
        tmp_60 = torch.nn.functional.gelu(conv1d_13);  conv1d_13 = None
        tmp_61 = torch.avg_pool1d(tmp_57, (2,), (2,), (0,), False, True);  tmp_57 = None
        tmp_62 = tmp_61[(Ellipsis, slice(None, 124, None))];  tmp_61 = None
        tmp_63 = tmp_60[(Ellipsis, slice(None, 124, None))];  tmp_60 = None
        tmp_64 = tmp_62 + tmp_63;  tmp_62 = tmp_63 = None
        tmp_65 = tmp_64.transpose(1, 2);  tmp_64 = None
        tmp_66 = torch.nn.functional.layer_norm(tmp_65, (768,), w_1, w_0, 1e-05);  tmp_65 = w_1 = w_0 = None
        tmp_67 = torch.nn.functional.dropout(tmp_66, 0.1, False, False);  tmp_66 = None
        tmp_68 = torch.rand([]);  tmp_68 = None
        return (tmp_67,)
        