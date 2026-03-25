import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21):
        linear = torch.nn.functional.linear(in_18, in_4, in_3);  in_18 = in_4 = in_3 = None
        tmp_18 = linear.view((64, 144, -1, 32));  linear = None
        tmp_19 = tmp_18.transpose(1, 2);  tmp_18 = None
        tmp_20 = in_19.transpose(-1, -2);  in_19 = None
        matmul = torch.matmul(in_21, tmp_20);  in_21 = tmp_20 = None
        tmp_22 = matmul / 5.656854249492381;  matmul = None
        tmp_23 = in_2.view(-1);  in_2 = None
        tmp_24 = in_5[tmp_23];  in_5 = tmp_23 = None
        tmp_25 = tmp_24.view(144, 144, -1);  tmp_24 = None
        tmp_26 = tmp_25.permute(2, 0, 1);  tmp_25 = None
        tmp_27 = tmp_26.contiguous();  tmp_26 = None
        tmp_28 = tmp_27.unsqueeze(0);  tmp_27 = None
        tmp_29 = tmp_22 + tmp_28;  tmp_22 = tmp_28 = None
        tmp_30 = tmp_29.view(1, 64, 4, 144, 144);  tmp_29 = None
        tmp_31 = in_17.unsqueeze(1);  in_17 = None
        tmp_32 = tmp_31.unsqueeze(0);  tmp_31 = None
        tmp_33 = tmp_30 + tmp_32;  tmp_30 = tmp_32 = None
        tmp_34 = tmp_33.view(-1, 4, 144, 144);  tmp_33 = None
        tmp_35 = torch.nn.functional.softmax(tmp_34, dim = -1);  tmp_34 = None
        tmp_36 = torch.nn.functional.dropout(tmp_35, 0.0, False, False);  tmp_35 = None
        matmul_1 = torch.matmul(tmp_36, tmp_19);  tmp_36 = tmp_19 = None
        tmp_38 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_39 = tmp_38.contiguous();  tmp_38 = None
        tmp_40 = tmp_39.view((64, 144, 128));  tmp_39 = None
        linear_1 = torch.nn.functional.linear(tmp_40, in_1, in_0);  tmp_40 = in_1 = in_0 = None
        tmp_42 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_43 = tmp_42.view(-1, 12, 12, 128);  tmp_42 = None
        tmp_44 = tmp_43.view(-1, 8, 8, 12, 12, 128);  tmp_43 = None
        tmp_45 = tmp_44.permute(0, 1, 3, 2, 4, 5);  tmp_44 = None
        tmp_46 = tmp_45.contiguous();  tmp_45 = None
        tmp_47 = tmp_46.view(-1, 96, 96, 128);  tmp_46 = None
        tmp_48 = torch.roll(tmp_47, shifts = (6, 6), dims = (1, 2));  tmp_47 = None
        tmp_49 = tmp_48.view(1, 9216, 128);  tmp_48 = None
        tmp_50 = in_20 + tmp_49;  in_20 = tmp_49 = None
        tmp_51 = torch.nn.functional.layer_norm(tmp_50, (128,), in_9, in_8, 1e-05);  in_9 = in_8 = None
        linear_2 = torch.nn.functional.linear(tmp_51, in_7, in_6);  tmp_51 = in_7 = in_6 = None
        tmp_53 = torch.nn.functional.gelu(linear_2);  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_53, in_11, in_10);  tmp_53 = in_11 = in_10 = None
        tmp_55 = torch.nn.functional.dropout(linear_3, 0.0, False, False);  linear_3 = None
        tmp_56 = tmp_50 + tmp_55;  tmp_50 = tmp_55 = None
        tmp_57 = tmp_56.view(1, 96, 96, 128);  tmp_56 = None
        tmp_58 = tmp_57[(slice(None, None, None), slice(0, None, 2), slice(0, None, 2), slice(None, None, None))]
        tmp_59 = tmp_57[(slice(None, None, None), slice(1, None, 2), slice(0, None, 2), slice(None, None, None))]
        tmp_60 = tmp_57[(slice(None, None, None), slice(0, None, 2), slice(1, None, 2), slice(None, None, None))]
        tmp_61 = tmp_57[(slice(None, None, None), slice(1, None, 2), slice(1, None, 2), slice(None, None, None))];  tmp_57 = None
        tmp_62 = torch.cat([tmp_58, tmp_59, tmp_60, tmp_61], -1);  tmp_58 = tmp_59 = tmp_60 = tmp_61 = None
        tmp_63 = tmp_62.view(1, -1, 512);  tmp_62 = None
        tmp_64 = torch.nn.functional.layer_norm(tmp_63, (512,), in_13, in_12, 1e-05);  tmp_63 = in_13 = in_12 = None
        linear_4 = torch.nn.functional.linear(tmp_64, in_14, None);  tmp_64 = in_14 = None
        tmp_66 = torch.nn.functional.layer_norm(linear_4, (256,), in_16, in_15, 1e-05);  in_16 = in_15 = None
        tmp_67 = tmp_66.view(1, 48, 48, 256);  tmp_66 = None
        tmp_68 = torch.nn.functional.pad(tmp_67, (0, 0, 0, 0, 0, 0), 'constant', None);  tmp_67 = None
        tmp_69 = tmp_68.view(1, 4, 12, 4, 12, 256);  tmp_68 = None
        tmp_70 = tmp_69.permute(0, 1, 3, 2, 4, 5);  tmp_69 = None
        tmp_71 = tmp_70.contiguous();  tmp_70 = None
        tmp_72 = tmp_71.view(-1, 12, 12, 256);  tmp_71 = None
        tmp_73 = tmp_72.view(-1, 144, 256);  tmp_72 = None
        return (tmp_73, linear_4)
        