import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18):
        linear = torch.nn.functional.linear(in_15, in_4, in_3);  in_15 = in_4 = in_3 = None
        tmp_15 = linear.view((4, 144, -1, 32));  linear = None
        tmp_16 = tmp_15.transpose(1, 2);  tmp_15 = None
        tmp_17 = in_16.transpose(-1, -2);  in_16 = None
        matmul = torch.matmul(in_18, tmp_17);  in_18 = tmp_17 = None
        tmp_19 = matmul / 5.656854249492381;  matmul = None
        tmp_20 = in_2.view(-1);  in_2 = None
        tmp_21 = in_5[tmp_20];  in_5 = tmp_20 = None
        tmp_22 = tmp_21.view(144, 144, -1);  tmp_21 = None
        tmp_23 = tmp_22.permute(2, 0, 1);  tmp_22 = None
        tmp_24 = tmp_23.contiguous();  tmp_23 = None
        tmp_25 = tmp_24.unsqueeze(0);  tmp_24 = None
        tmp_26 = tmp_19 + tmp_25;  tmp_19 = tmp_25 = None
        tmp_27 = tmp_26.view(1, 4, 24, 144, 144);  tmp_26 = None
        tmp_28 = in_14.unsqueeze(1);  in_14 = None
        tmp_29 = tmp_28.unsqueeze(0);  tmp_28 = None
        tmp_30 = tmp_27 + tmp_29;  tmp_27 = tmp_29 = None
        tmp_31 = tmp_30.view(-1, 24, 144, 144);  tmp_30 = None
        tmp_32 = torch.nn.functional.softmax(tmp_31, dim = -1);  tmp_31 = None
        tmp_33 = torch.nn.functional.dropout(tmp_32, 0.0, False, False);  tmp_32 = None
        matmul_1 = torch.matmul(tmp_33, tmp_16);  tmp_33 = tmp_16 = None
        tmp_35 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_36 = tmp_35.contiguous();  tmp_35 = None
        tmp_37 = tmp_36.view((4, 144, 768));  tmp_36 = None
        linear_1 = torch.nn.functional.linear(tmp_37, in_1, in_0);  tmp_37 = in_1 = in_0 = None
        tmp_39 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_40 = tmp_39.view(-1, 12, 12, 768);  tmp_39 = None
        tmp_41 = tmp_40.view(-1, 2, 2, 12, 12, 768);  tmp_40 = None
        tmp_42 = tmp_41.permute(0, 1, 3, 2, 4, 5);  tmp_41 = None
        tmp_43 = tmp_42.contiguous();  tmp_42 = None
        tmp_44 = tmp_43.view(-1, 24, 24, 768);  tmp_43 = None
        tmp_45 = torch.roll(tmp_44, shifts = (6, 6), dims = (1, 2));  tmp_44 = None
        tmp_46 = tmp_45.view(1, 576, 768);  tmp_45 = None
        tmp_47 = in_17 + tmp_46;  in_17 = tmp_46 = None
        tmp_48 = torch.nn.functional.layer_norm(tmp_47, (768,), in_9, in_8, 1e-05);  in_9 = in_8 = None
        linear_2 = torch.nn.functional.linear(tmp_48, in_7, in_6);  tmp_48 = in_7 = in_6 = None
        tmp_50 = torch.nn.functional.gelu(linear_2);  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_50, in_11, in_10);  tmp_50 = in_11 = in_10 = None
        tmp_52 = torch.nn.functional.dropout(linear_3, 0.0, False, False);  linear_3 = None
        tmp_53 = tmp_47 + tmp_52;  tmp_47 = tmp_52 = None
        tmp_54 = torch.nn.functional.layer_norm(tmp_53, (768,), in_13, in_12, 1e-05);  in_13 = in_12 = None
        tmp_55 = tmp_54.view(1, 24, 24, 768);  tmp_54 = None
        tmp_56 = torch.nn.functional.pad(tmp_55, (0, 0, 0, 0, 0, 0), 'constant', None);  tmp_55 = None
        tmp_57 = tmp_56.view(1, 2, 12, 2, 12, 768);  tmp_56 = None
        tmp_58 = tmp_57.permute(0, 1, 3, 2, 4, 5);  tmp_57 = None
        tmp_59 = tmp_58.contiguous();  tmp_58 = None
        tmp_60 = tmp_59.view(-1, 12, 12, 768);  tmp_59 = None
        tmp_61 = tmp_60.view(-1, 144, 768);  tmp_60 = None
        return (tmp_61, tmp_53)
        