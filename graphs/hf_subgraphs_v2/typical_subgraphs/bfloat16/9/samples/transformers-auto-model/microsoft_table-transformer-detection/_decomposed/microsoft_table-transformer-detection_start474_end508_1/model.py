import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1, in_2, in_3):
        tmp_10 = torch.nn.functional.relu(in_2, inplace = False);  in_2 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, p = 0.0, training = False);  tmp_10 = None
        linear = torch.nn.functional.linear(tmp_11, w_1, w_0);  tmp_11 = w_1 = w_0 = None
        tmp_13 = torch.nn.functional.dropout(linear, p = 0.1, training = False);  linear = None
        tmp_14 = in_1 + tmp_13;  in_1 = tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (256,), w_3, w_2, 1e-05);  w_3 = w_2 = None
        tmp_16 = tmp_15 + in_3;  in_3 = None
        linear_1 = torch.nn.functional.linear(tmp_16, w_7, w_6);  w_7 = w_6 = None
        tmp_18 = linear_1 * 0.1767766952966369;  linear_1 = None
        linear_2 = torch.nn.functional.linear(tmp_16, w_5, w_4);  tmp_16 = w_5 = w_4 = None
        tmp_20 = linear_2.view(1, -1, 8, 32);  linear_2 = None
        tmp_21 = tmp_20.transpose(1, 2);  tmp_20 = None
        tmp_22 = tmp_21.contiguous();  tmp_21 = None
        linear_3 = torch.nn.functional.linear(tmp_15, w_9, w_8);  tmp_15 = w_9 = w_8 = None
        tmp_24 = linear_3.view(1, -1, 8, 32);  linear_3 = None
        tmp_25 = tmp_24.transpose(1, 2);  tmp_24 = None
        tmp_26 = tmp_25.contiguous();  tmp_25 = None
        tmp_27 = tmp_18.view(1, 625, 8, 32);  tmp_18 = None
        tmp_28 = tmp_27.transpose(1, 2);  tmp_27 = None
        tmp_29 = tmp_28.contiguous();  tmp_28 = None
        tmp_30 = tmp_29.view(8, -1, 32);  tmp_29 = None
        tmp_31 = tmp_22.view(8, -1, 32);  tmp_22 = None
        tmp_32 = tmp_26.view(8, -1, 32);  tmp_26 = None
        tmp_33 = tmp_31.transpose(1, 2);  tmp_31 = None
        bmm = torch.bmm(tmp_30, tmp_33);  tmp_30 = tmp_33 = None
        tmp_35 = bmm.view(1, 8, 625, 625);  bmm = None
        tmp_36 = tmp_35 + in_0;  tmp_35 = in_0 = None
        tmp_37 = tmp_36.view(8, 625, 625);  tmp_36 = None
        tmp_38 = torch.nn.functional.softmax(tmp_37, dim = -1);  tmp_37 = None
        tmp_39 = torch.nn.functional.dropout(tmp_38, p = 0.0, training = False);  tmp_38 = None
        bmm_1 = torch.bmm(tmp_39, tmp_32);  tmp_39 = tmp_32 = None
        tmp_41 = bmm_1.view(1, 8, 625, 32);  bmm_1 = None
        tmp_42 = tmp_41.transpose(1, 2);  tmp_41 = None
        tmp_43 = tmp_42.reshape(1, 625, 256);  tmp_42 = None
        return (tmp_43, tmp_14)
        