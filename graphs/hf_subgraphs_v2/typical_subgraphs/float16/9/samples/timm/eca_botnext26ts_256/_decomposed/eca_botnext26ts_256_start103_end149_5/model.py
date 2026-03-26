import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, in_0 : torch.Tensor):
        tmp_7 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        conv2d = torch.conv2d(tmp_7, w_6, None, (1, 1), (0, 0), (1, 1), 1);  tmp_7 = w_6 = None
        split = torch.functional.split(conv2d, [64, 64, 256], dim = 1);  conv2d = None
        tmp_10 = split[0]
        tmp_11 = split[1]
        tmp_12 = split[2];  split = None
        tmp_13 = tmp_10.reshape(4, 16, -1);  tmp_10 = None
        tmp_14 = tmp_13.transpose(-1, -2);  tmp_13 = None
        tmp_15 = tmp_11.reshape(4, 16, -1);  tmp_11 = None
        tmp_16 = tmp_12.reshape(4, 64, -1);  tmp_12 = None
        tmp_17 = tmp_16.transpose(-1, -2);  tmp_16 = None
        matmul = tmp_14 @ tmp_15;  tmp_15 = None
        tmp_19 = matmul * 0.25;  matmul = None
        tmp_20 = tmp_14.reshape(4, 16, 16, -1);  tmp_14 = None
        tmp_21 = w_5.transpose(-1, -2);  w_5 = None
        matmul_1 = tmp_20 @ tmp_21;  tmp_21 = None
        tmp_23 = matmul_1.reshape(-1, 16, 31);  matmul_1 = None
        tmp_24 = torch.nn.functional.pad(tmp_23, [0, 1], 'constant', None);  tmp_23 = None
        tmp_25 = tmp_24.flatten(1);  tmp_24 = None
        tmp_26 = torch.nn.functional.pad(tmp_25, [0, 15], 'constant', None);  tmp_25 = None
        tmp_27 = tmp_26.reshape(-1, 17, 31);  tmp_26 = None
        tmp_28 = tmp_27[(slice(None, None, None), slice(None, 16, None), slice(15, None, None))];  tmp_27 = None
        tmp_29 = tmp_28.reshape(4, 16, 1, 16, 16);  tmp_28 = None
        tmp_30 = tmp_29.expand(-1, -1, 16, -1, -1);  tmp_29 = None
        tmp_31 = tmp_30.permute((0, 1, 3, 2, 4));  tmp_30 = None
        tmp_32 = tmp_20.transpose(1, 2);  tmp_20 = None
        tmp_33 = w_4.transpose(-1, -2);  w_4 = None
        matmul_2 = tmp_32 @ tmp_33;  tmp_32 = tmp_33 = None
        tmp_35 = matmul_2.reshape(-1, 16, 31);  matmul_2 = None
        tmp_36 = torch.nn.functional.pad(tmp_35, [0, 1], 'constant', None);  tmp_35 = None
        tmp_37 = tmp_36.flatten(1);  tmp_36 = None
        tmp_38 = torch.nn.functional.pad(tmp_37, [0, 15], 'constant', None);  tmp_37 = None
        tmp_39 = tmp_38.reshape(-1, 17, 31);  tmp_38 = None
        tmp_40 = tmp_39[(slice(None, None, None), slice(None, 16, None), slice(15, None, None))];  tmp_39 = None
        tmp_41 = tmp_40.reshape(4, 16, 1, 16, 16);  tmp_40 = None
        tmp_42 = tmp_41.expand(-1, -1, 16, -1, -1);  tmp_41 = None
        tmp_43 = tmp_42.permute((0, 3, 1, 4, 2));  tmp_42 = None
        tmp_44 = tmp_43 + tmp_31;  tmp_43 = tmp_31 = None
        tmp_45 = tmp_44.reshape(4, 256, 256);  tmp_44 = None
        tmp_46 = tmp_19 + tmp_45;  tmp_19 = tmp_45 = None
        tmp_47 = tmp_46.softmax(dim = -1);  tmp_46 = None
        matmul_3 = tmp_47 @ tmp_17;  tmp_47 = tmp_17 = None
        tmp_49 = matmul_3.transpose(-1, -2);  matmul_3 = None
        tmp_50 = tmp_49.reshape(1, 256, 16, 16);  tmp_49 = None
        tmp_51 = torch.nn.functional.batch_norm(tmp_50, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  tmp_50 = w_0 = w_1 = w_3 = w_2 = None
        tmp_52 = torch.nn.functional.silu(tmp_51, inplace = True);  tmp_51 = None
        return (tmp_52,)
        