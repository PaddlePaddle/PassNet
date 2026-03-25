import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0, w_1, w_2, w_3, in_1, in_2):
        tmp_5 = in_2 * 0.5;  in_2 = None
        linear = torch.nn.functional.linear(in_0, w_1, w_0);  w_1 = w_0 = None
        linear_1 = torch.nn.functional.linear(in_0, w_3, w_2);  in_0 = w_3 = w_2 = None
        tmp_8 = linear.view(1, -1, 4, 4);  linear = None
        tmp_9 = tmp_8.transpose(1, 2);  tmp_8 = None
        tmp_10 = linear_1.view(1, -1, 4, 4);  linear_1 = None
        tmp_11 = tmp_10.transpose(1, 2);  tmp_10 = None
        tmp_12 = tmp_5.view(1, 22, 4, 4);  tmp_5 = None
        tmp_13 = tmp_12.transpose(1, 2);  tmp_12 = None
        tmp_14 = tmp_13.reshape(4, -1, 4);  tmp_13 = None
        tmp_15 = tmp_9.reshape(4, -1, 4)
        tmp_16 = tmp_11.reshape(4, -1, 4)
        tmp_17 = tmp_15.transpose(1, 2);  tmp_15 = None
        bmm = torch.bmm(tmp_14, tmp_17);  tmp_14 = tmp_17 = None
        tmp_19 = bmm.view(1, 4, 22, 22);  bmm = None
        tmp_20 = tmp_19 + in_1;  tmp_19 = in_1 = None
        tmp_21 = tmp_20.view(4, 22, 22);  tmp_20 = None
        tmp_22 = torch.nn.functional.softmax(tmp_21, dim = -1);  tmp_21 = None
        tmp_23 = torch.nn.functional.dropout(tmp_22, p = 0.1, training = False);  tmp_22 = None
        bmm_1 = torch.bmm(tmp_23, tmp_16);  tmp_23 = tmp_16 = None
        tmp_25 = bmm_1.view(1, 4, 22, 4);  bmm_1 = None
        tmp_26 = tmp_25.transpose(1, 2);  tmp_25 = None
        tmp_27 = tmp_26.reshape(1, 22, 16);  tmp_26 = None
        return (tmp_27, tmp_9, tmp_11)
        