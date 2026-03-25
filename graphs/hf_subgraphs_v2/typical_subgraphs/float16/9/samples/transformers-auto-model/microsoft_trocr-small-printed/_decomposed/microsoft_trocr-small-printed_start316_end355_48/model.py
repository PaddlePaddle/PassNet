import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, in_1, in_2):
        tmp_17 = torch.nn.functional.gelu(in_2);  in_2 = None
        linear = torch.nn.functional.linear(tmp_17, w_11, w_10);  tmp_17 = w_11 = w_10 = None
        tmp_19 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_20 = tmp_19 + in_1;  tmp_19 = in_1 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (384,), w_13, w_12, 1e-12);  tmp_20 = w_13 = w_12 = None
        tmp_22 = tmp_21[(slice(None, None, None), 0)]
        linear_1 = torch.nn.functional.linear(tmp_22, w_15, w_14);  tmp_22 = w_15 = w_14 = None
        tmp_24 = torch.tanh(linear_1);  linear_1 = tmp_24 = None
        tmp_25 = in_0.view(-1, 1);  in_0 = None
        tmp_26 = torch.nn.functional.embedding(tmp_25, w_1, 1, None, 2.0, False, False);  tmp_25 = w_1 = None
        tmp_27 = tmp_26 * 16.0;  tmp_26 = None
        tmp_28 = torch.arange(0, 1, dtype = torch.int64, device = device(type='cuda', index=0))
        tmp_29 = tmp_28.expand(1, -1);  tmp_28 = None
        tmp_30 = tmp_29 + 2;  tmp_29 = None
        tmp_31 = torch.nn.functional.embedding(tmp_30, w_0, None, None, 2.0, False, False);  tmp_30 = w_0 = None
        tmp_32 = tmp_27 + tmp_31;  tmp_27 = tmp_31 = None
        tmp_33 = torch.nn.functional.layer_norm(tmp_32, (256,), w_3, w_2, 1e-05);  tmp_32 = w_3 = w_2 = None
        tmp_34 = torch.nn.functional.dropout(tmp_33, p = 0.1, training = False);  tmp_33 = None
        linear_2 = torch.nn.functional.linear(tmp_34, w_7, w_6);  w_7 = w_6 = None
        tmp_36 = linear_2 * 0.1767766952966369;  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_34, w_5, w_4);  w_5 = w_4 = None
        linear_4 = torch.nn.functional.linear(tmp_34, w_9, w_8);  w_9 = w_8 = None
        tmp_39 = linear_3.view(1, -1, 8, 32);  linear_3 = None
        tmp_40 = tmp_39.transpose(1, 2);  tmp_39 = None
        tmp_41 = linear_4.view(1, -1, 8, 32);  linear_4 = None
        tmp_42 = tmp_41.transpose(1, 2);  tmp_41 = None
        tmp_43 = tmp_36.view(1, 1, 8, 32);  tmp_36 = None
        tmp_44 = tmp_43.transpose(1, 2);  tmp_43 = None
        tmp_45 = tmp_44.reshape(8, -1, 32);  tmp_44 = None
        tmp_46 = tmp_40.reshape(8, -1, 32);  tmp_40 = None
        tmp_47 = tmp_42.reshape(8, -1, 32);  tmp_42 = None
        tmp_48 = tmp_46.transpose(1, 2);  tmp_46 = None
        bmm = torch.bmm(tmp_45, tmp_48);  tmp_45 = tmp_48 = None
        tmp_50 = torch.nn.functional.softmax(bmm, dim = -1);  bmm = None
        tmp_51 = torch.nn.functional.dropout(tmp_50, p = 0.0, training = False);  tmp_50 = None
        bmm_1 = torch.bmm(tmp_51, tmp_47);  tmp_51 = tmp_47 = None
        tmp_53 = bmm_1.view(1, 8, 1, 32);  bmm_1 = None
        tmp_54 = tmp_53.transpose(1, 2);  tmp_53 = None
        tmp_55 = tmp_54.reshape(1, 1, 256);  tmp_54 = None
        return (tmp_55, tmp_34, tmp_21)
        