import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0, in_1, in_2, in_3):
        tmp_12 = torch.nn.functional.relu(in_1);  in_1 = None
        linear = torch.nn.functional.linear(tmp_12, w_11, w_10);  tmp_12 = w_11 = w_10 = None
        tmp_14 = in_2 * linear;  in_2 = linear = None
        linear_1 = torch.nn.functional.linear(in_0, w_5, w_4);  w_5 = w_4 = None
        linear_2 = torch.nn.functional.linear(in_3, w_7, w_6);  w_7 = w_6 = None
        linear_3 = torch.nn.functional.linear(in_0, w_1, w_0);  w_1 = w_0 = None
        linear_4 = torch.nn.functional.linear(in_3, w_3, w_2);  in_3 = w_3 = w_2 = None
        linear_5 = torch.nn.functional.linear(in_0, w_9, w_8);  in_0 = w_9 = w_8 = None
        tmp_20 = linear_1 + linear_2;  linear_1 = linear_2 = None
        tmp_21 = linear_3 + linear_4;  linear_3 = linear_4 = None
        tmp_22 = tmp_20 * 0.1767766952966369;  tmp_20 = None
        tmp_23 = tmp_21.view(1, -1, 8, 32);  tmp_21 = None
        tmp_24 = tmp_23.transpose(1, 2);  tmp_23 = None
        tmp_25 = tmp_24.contiguous();  tmp_24 = None
        tmp_26 = linear_5.view(1, -1, 8, 32);  linear_5 = None
        tmp_27 = tmp_26.transpose(1, 2);  tmp_26 = None
        tmp_28 = tmp_27.contiguous();  tmp_27 = None
        tmp_29 = tmp_22.view(1, 300, 8, 32);  tmp_22 = None
        tmp_30 = tmp_29.transpose(1, 2);  tmp_29 = None
        tmp_31 = tmp_30.contiguous();  tmp_30 = None
        tmp_32 = tmp_31.view(8, -1, 32);  tmp_31 = None
        tmp_33 = tmp_25.view(8, -1, 32);  tmp_25 = None
        tmp_34 = tmp_28.view(8, -1, 32);  tmp_28 = None
        tmp_35 = tmp_33.transpose(1, 2);  tmp_33 = None
        bmm = torch.bmm(tmp_32, tmp_35);  tmp_32 = tmp_35 = None
        tmp_37 = torch.nn.functional.softmax(bmm, dim = -1);  bmm = None
        tmp_38 = tmp_37.view(1, 8, 300, 300);  tmp_37 = None
        tmp_39 = tmp_38.view(8, 300, 300)
        tmp_40 = torch.nn.functional.dropout(tmp_39, p = 0.0, training = False);  tmp_39 = None
        bmm_1 = torch.bmm(tmp_40, tmp_34);  tmp_40 = tmp_34 = None
        tmp_42 = bmm_1.view(1, 8, 300, 32);  bmm_1 = None
        tmp_43 = tmp_42.transpose(1, 2);  tmp_42 = None
        tmp_44 = tmp_43.reshape(1, 300, 256);  tmp_43 = None
        return (tmp_44, tmp_38, tmp_14)
        