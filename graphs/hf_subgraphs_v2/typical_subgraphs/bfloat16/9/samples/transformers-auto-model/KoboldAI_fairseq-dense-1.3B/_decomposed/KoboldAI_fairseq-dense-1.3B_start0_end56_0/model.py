import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor):
        tmp_11 = torch.full((9, 9), -3.4028234663852886e+38, device = device(type='cuda', index=0))
        tmp_12 = torch.arange(9, device = device(type='cuda', index=0))
        tmp_13 = tmp_12 + 1
        tmp_14 = tmp_13.view(9, 1);  tmp_13 = None
        tmp_15 = tmp_12 < tmp_14;  tmp_12 = tmp_14 = None
        tmp_16 = tmp_11.masked_fill_(tmp_15, 0);  tmp_15 = tmp_16 = None
        tmp_17 = tmp_11.to(torch.float32);  tmp_11 = None
        tmp_18 = tmp_17[(None, None, slice(None, None, None), slice(None, None, None))];  tmp_17 = None
        tmp_19 = tmp_18.expand(1, 1, 9, 9);  tmp_18 = None
        tmp_20 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_21 = tmp_20.expand(1, 1, 9, 9);  tmp_20 = None
        tmp_22 = tmp_21.to(torch.float32);  tmp_21 = None
        tmp_23 = torch.tensor(1.0, dtype = torch.float32)
        tmp_24 = tmp_23 - tmp_22;  tmp_23 = tmp_22 = None
        tmp_25 = tmp_24.to(torch.bool)
        tmp_26 = tmp_24.masked_fill(tmp_25, -3.4028234663852886e+38);  tmp_24 = tmp_25 = None
        tmp_27 = tmp_26.to(device(type='cuda', index=0));  tmp_26 = None
        tmp_28 = tmp_27.bool();  tmp_27 = None
        tmp_29 = tmp_19.masked_fill(tmp_28, -3.4028234663852886e+38);  tmp_19 = tmp_28 = None
        tmp_30 = torch.arange(0, 9, dtype = torch.int64, device = device(type='cuda', index=0))
        tmp_31 = tmp_30.unsqueeze(0);  tmp_30 = None
        tmp_31 += 2;  tmp_32 = tmp_31;  tmp_31 = None
        tmp_33 = tmp_32.view(-1);  tmp_32 = None
        tmp_34 = w_0.index_select(0, tmp_33);  w_0 = tmp_33 = None
        tmp_35 = tmp_34.view(1, 9, 2048);  tmp_34 = None
        tmp_36 = tmp_35.detach();  tmp_35 = None
        tmp_37 = tmp_36.to(device(type='cuda', index=0));  tmp_36 = None
        tmp_38 = in_1 + tmp_37;  in_1 = tmp_37 = None
        tmp_39 = torch.nn.functional.dropout(tmp_38, p = 0.1, training = False);  tmp_38 = None
        tmp_40 = torch.nn.functional.layer_norm(tmp_39, (2048,), w_2, w_1, 1e-05);  w_2 = w_1 = None
        linear = torch.nn.functional.linear(tmp_40, w_6, w_5);  w_6 = w_5 = None
        tmp_42 = linear * 0.125;  linear = None
        linear_1 = torch.nn.functional.linear(tmp_40, w_4, w_3);  w_4 = w_3 = None
        linear_2 = torch.nn.functional.linear(tmp_40, w_8, w_7);  tmp_40 = w_8 = w_7 = None
        tmp_45 = linear_1.view(1, 9, -1, 64);  linear_1 = None
        tmp_46 = tmp_45.transpose(1, 2);  tmp_45 = None
        tmp_47 = linear_2.view(1, 9, -1, 64);  linear_2 = None
        tmp_48 = tmp_47.transpose(1, 2);  tmp_47 = None
        tmp_49 = tmp_42.view(1, 9, 32, 64);  tmp_42 = None
        tmp_50 = tmp_49.transpose(1, 2);  tmp_49 = None
        tmp_51 = tmp_50.reshape(32, -1, 64);  tmp_50 = None
        tmp_52 = tmp_46.reshape(32, -1, 64)
        tmp_53 = tmp_48.reshape(32, -1, 64)
        tmp_54 = tmp_52.transpose(1, 2);  tmp_52 = None
        bmm = torch.bmm(tmp_51, tmp_54);  tmp_51 = tmp_54 = None
        tmp_56 = bmm.view(1, 32, 9, 9);  bmm = None
        tmp_57 = tmp_56 + tmp_29;  tmp_56 = None
        tmp_58 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
        tmp_59 = torch.max(tmp_57, tmp_58);  tmp_57 = tmp_58 = None
        tmp_60 = tmp_59.view(32, 9, 9);  tmp_59 = None
        tmp_61 = torch.nn.functional.softmax(tmp_60, dim = -1);  tmp_60 = None
        tmp_62 = torch.nn.functional.dropout(tmp_61, p = 0.1, training = False);  tmp_61 = None
        to_5 = tmp_62.to(torch.bfloat16);  tmp_62 = None
        bmm_1 = torch.bmm(to_5, tmp_53);  to_5 = tmp_53 = None
        tmp_64 = bmm_1.view(1, 32, 9, 64);  bmm_1 = None
        tmp_65 = tmp_64.transpose(1, 2);  tmp_64 = None
        tmp_66 = tmp_65.reshape(1, 9, 2048);  tmp_65 = None
        return (tmp_66, tmp_29, tmp_39, tmp_46, tmp_48)
        