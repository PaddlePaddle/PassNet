import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1, in_2):
        tmp_10 = torch.nn.functional.gelu(in_0, approximate = 'none');  in_0 = None
        tmp_11 = tmp_10.flatten(2);  tmp_10 = None
        tmp_12 = tmp_11.transpose(1, 2);  tmp_11 = None
        tmp_13 = tmp_12.contiguous();  tmp_12 = None
        tmp_14 = in_2 + tmp_13;  in_2 = tmp_13 = None
        tmp_15 = tmp_14.permute(0, 2, 1);  tmp_14 = None
        tmp_16 = tmp_15.view(1, 32, 64, 48);  tmp_15 = None
        tmp_17 = in_1.view(1, 64, -1);  in_1 = None
        tmp_18 = tmp_17.permute(0, 2, 1);  tmp_17 = None
        tmp_19 = torch.nn.functional.layer_norm(tmp_18, (64,), w_7, w_6, 1e-06);  w_7 = w_6 = None
        tmp_20 = tmp_19.view(1, 32, 24, 64);  tmp_19 = None
        tmp_21 = torch.nn.functional.pad(tmp_20, (0, 0, 2, 2, 1, 2), 'constant', None);  tmp_20 = None
        tmp_22 = tmp_21.view(1, 5, 7, 4, 7, 64);  tmp_21 = None
        tmp_23 = tmp_22.permute(0, 1, 3, 2, 4, 5);  tmp_22 = None
        tmp_24 = tmp_23.reshape(-1, 49, 64);  tmp_23 = None
        linear = torch.nn.functional.linear(tmp_24, w_4, w_3);  tmp_24 = w_4 = w_3 = None
        tmp_26 = linear.reshape(20, 49, 3, 2, 32);  linear = None
        tmp_27 = tmp_26.permute(2, 0, 3, 1, 4);  tmp_26 = None
        tmp_28 = tmp_27[0]
        tmp_29 = tmp_27[1]
        tmp_30 = tmp_27[2];  tmp_27 = None
        tmp_31 = tmp_28 * 0.1767766952966369;  tmp_28 = None
        tmp_32 = tmp_29.transpose(-2, -1);  tmp_29 = None
        matmul = tmp_31 @ tmp_32;  tmp_31 = tmp_32 = None
        tmp_34 = w_0.view(-1);  w_0 = None
        tmp_35 = w_5[tmp_34];  w_5 = tmp_34 = None
        tmp_36 = tmp_35.view(49, 49, -1);  tmp_35 = None
        tmp_37 = tmp_36.permute(2, 0, 1);  tmp_36 = None
        tmp_38 = tmp_37.contiguous();  tmp_37 = None
        tmp_39 = tmp_38.unsqueeze(0);  tmp_38 = None
        tmp_40 = matmul + tmp_39;  matmul = tmp_39 = None
        tmp_41 = torch.nn.functional.softmax(tmp_40, -1, _stacklevel = 5);  tmp_40 = None
        tmp_42 = torch.nn.functional.dropout(tmp_41, 0.0, False, False);  tmp_41 = None
        matmul_1 = tmp_42 @ tmp_30;  tmp_42 = tmp_30 = None
        tmp_44 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_45 = tmp_44.reshape(20, 49, 64);  tmp_44 = None
        linear_1 = torch.nn.functional.linear(tmp_45, w_2, w_1);  tmp_45 = w_2 = w_1 = None
        tmp_47 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_48 = tmp_47.reshape(1, 5, 4, 7, 7, 64);  tmp_47 = None
        tmp_49 = tmp_48.permute(0, 1, 3, 2, 4, 5);  tmp_48 = None
        tmp_50 = tmp_49.reshape(1, 35, 28, 64);  tmp_49 = None
        tmp_51 = tmp_50[(slice(None, None, None), slice(1, 33, None), slice(2, 26, None))];  tmp_50 = None
        tmp_52 = tmp_51.reshape(1, 768, 64);  tmp_51 = None
        tmp_53 = tmp_18 + tmp_52;  tmp_18 = tmp_52 = None
        tmp_54 = torch.nn.functional.layer_norm(tmp_53, (64,), w_9, w_8, 1e-06);  w_9 = w_8 = None
        tmp_55 = tmp_54.transpose(1, 2);  tmp_54 = None
        tmp_56 = tmp_55.reshape(1, 64, 32, 24);  tmp_55 = None
        return (tmp_16, tmp_53, tmp_56)
        