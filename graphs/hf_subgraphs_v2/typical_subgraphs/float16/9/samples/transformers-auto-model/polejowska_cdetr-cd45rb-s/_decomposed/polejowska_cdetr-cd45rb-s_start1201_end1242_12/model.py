import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1, in_2, in_3, in_4, in_5):
        linear = torch.nn.functional.linear(in_1, w_1, w_0);  w_1 = w_0 = None
        linear_1 = torch.nn.functional.linear(in_1, w_9, w_8);  in_1 = w_9 = w_8 = None
        linear_2 = torch.nn.functional.linear(in_2, w_3, w_2);  in_2 = w_3 = w_2 = None
        linear_3 = torch.nn.functional.linear(in_4, w_5, w_4);  in_4 = w_5 = w_4 = None
        tmp_14 = in_3 + linear_3;  in_3 = linear_3 = None
        tmp_15 = linear + linear_2;  linear = None
        tmp_16 = tmp_14.view(1, 300, 8, 32);  tmp_14 = None
        linear_4 = torch.nn.functional.linear(in_5, w_7, w_6);  in_5 = w_7 = w_6 = None
        tmp_18 = linear_4.view(1, 300, 8, 32);  linear_4 = None
        tmp_19 = torch.cat([tmp_16, tmp_18], dim = 3);  tmp_16 = tmp_18 = None
        tmp_20 = tmp_19.view(1, 300, 512);  tmp_19 = None
        tmp_21 = tmp_15.view(1, 625, 8, 32);  tmp_15 = None
        tmp_22 = linear_2.view(1, 625, 8, 32);  linear_2 = None
        tmp_23 = torch.cat([tmp_21, tmp_22], dim = 3);  tmp_21 = tmp_22 = None
        tmp_24 = tmp_23.view(1, 625, 512);  tmp_23 = None
        tmp_25 = tmp_20 * 0.125;  tmp_20 = None
        tmp_26 = tmp_24.view(1, -1, 8, 64);  tmp_24 = None
        tmp_27 = tmp_26.transpose(1, 2);  tmp_26 = None
        tmp_28 = tmp_27.contiguous();  tmp_27 = None
        tmp_29 = linear_1.view(1, -1, 8, 32);  linear_1 = None
        tmp_30 = tmp_29.transpose(1, 2);  tmp_29 = None
        tmp_31 = tmp_30.contiguous();  tmp_30 = None
        tmp_32 = tmp_25.view(1, 300, 8, 64);  tmp_25 = None
        tmp_33 = tmp_32.transpose(1, 2);  tmp_32 = None
        tmp_34 = tmp_33.contiguous();  tmp_33 = None
        tmp_35 = tmp_34.view(8, -1, 64);  tmp_34 = None
        tmp_36 = tmp_28.view(8, -1, 64);  tmp_28 = None
        tmp_37 = tmp_31.view(8, -1, 32);  tmp_31 = None
        tmp_38 = tmp_36.transpose(1, 2);  tmp_36 = None
        bmm = torch.bmm(tmp_35, tmp_38);  tmp_35 = tmp_38 = None
        tmp_40 = bmm.view(1, 8, 300, 625);  bmm = None
        tmp_41 = tmp_40 + in_0;  tmp_40 = in_0 = None
        tmp_42 = tmp_41.view(8, 300, 625);  tmp_41 = None
        tmp_43 = torch.nn.functional.softmax(tmp_42, dim = -1);  tmp_42 = None
        tmp_44 = tmp_43.view(1, 8, 300, 625);  tmp_43 = None
        tmp_45 = tmp_44.view(8, 300, 625)
        tmp_46 = torch.nn.functional.dropout(tmp_45, p = 0.0, training = False);  tmp_45 = None
        bmm_1 = torch.bmm(tmp_46, tmp_37);  tmp_46 = tmp_37 = None
        tmp_48 = bmm_1.view(1, 8, 300, 32);  bmm_1 = None
        tmp_49 = tmp_48.transpose(1, 2);  tmp_48 = None
        tmp_50 = tmp_49.reshape(1, 300, 256);  tmp_49 = None
        return (tmp_50, tmp_44)
        