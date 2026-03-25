import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, in_0, in_1, in_2, in_3):
        tmp_11 = torch.nn.functional.relu(in_2, inplace = False);  in_2 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, p = 0.0, training = False);  tmp_11 = None
        linear = torch.nn.functional.linear(tmp_12, w_1, w_0);  tmp_12 = w_1 = w_0 = None
        tmp_14 = torch.nn.functional.dropout(linear, p = 0.1, training = False);  linear = None
        tmp_15 = in_1 + tmp_14;  in_1 = tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (256,), w_3, w_2, 1e-05);  tmp_15 = w_3 = w_2 = None
        tmp_17 = tmp_16 + in_3;  in_3 = None
        linear_1 = torch.nn.functional.linear(tmp_17, w_7, w_6);  w_7 = w_6 = None
        item = w_10.item();  w_10 = None
        tmp_20 = linear_1 * item;  linear_1 = item = None
        linear_2 = torch.nn.functional.linear(tmp_17, w_5, w_4);  tmp_17 = w_5 = w_4 = None
        tmp_22 = linear_2.view(1, -1, 4, 64);  linear_2 = None
        tmp_23 = tmp_22.transpose(1, 2);  tmp_22 = None
        tmp_24 = tmp_23.contiguous();  tmp_23 = None
        linear_3 = torch.nn.functional.linear(tmp_16, w_9, w_8);  w_9 = w_8 = None
        tmp_26 = linear_3.view(1, -1, 4, 64);  linear_3 = None
        tmp_27 = tmp_26.transpose(1, 2);  tmp_26 = None
        tmp_28 = tmp_27.contiguous();  tmp_27 = None
        tmp_29 = tmp_20.view(1, 625, 4, 64);  tmp_20 = None
        tmp_30 = tmp_29.transpose(1, 2);  tmp_29 = None
        tmp_31 = tmp_30.contiguous();  tmp_30 = None
        tmp_32 = tmp_31.view(4, -1, 64);  tmp_31 = None
        tmp_33 = tmp_24.view(4, -1, 64);  tmp_24 = None
        tmp_34 = tmp_28.view(4, -1, 64);  tmp_28 = None
        tmp_35 = tmp_33.transpose(1, 2);  tmp_33 = None
        bmm = torch.bmm(tmp_32, tmp_35);  tmp_32 = tmp_35 = None
        tmp_37 = bmm.view(1, 4, 625, 625);  bmm = None
        tmp_38 = tmp_37 + in_0;  tmp_37 = in_0 = None
        tmp_39 = tmp_38.view(4, 625, 625);  tmp_38 = None
        tmp_40 = torch.nn.functional.softmax(tmp_39, dim = -1);  tmp_39 = None
        tmp_41 = tmp_40.view(1, 4, 625, 625);  tmp_40 = None
        tmp_42 = tmp_41.view(4, 625, 625)
        tmp_43 = torch.nn.functional.dropout(tmp_42, p = 0.0, training = False);  tmp_42 = None
        bmm_1 = torch.bmm(tmp_43, tmp_34);  tmp_43 = tmp_34 = None
        tmp_45 = bmm_1.view(1, 4, 625, 64);  bmm_1 = None
        tmp_46 = tmp_45.transpose(1, 2);  tmp_45 = None
        tmp_47 = tmp_46.reshape(1, 625, 256);  tmp_46 = None
        return (tmp_47, tmp_41, tmp_16)
        