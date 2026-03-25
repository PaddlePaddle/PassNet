import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, in_0, in_1, in_2):
        tmp_4 = in_2 * 0.5;  in_2 = None
        linear = torch.nn.functional.linear(in_1, w_1, w_0);  w_1 = w_0 = None
        linear_1 = torch.nn.functional.linear(in_1, w_3, w_2);  in_1 = w_3 = w_2 = None
        tmp_7 = linear.view(1, -1, 4, 4);  linear = None
        tmp_8 = tmp_7.transpose(1, 2);  tmp_7 = None
        tmp_9 = linear_1.view(1, -1, 4, 4);  linear_1 = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        tmp_11 = tmp_4.view(1, 22, 4, 4);  tmp_4 = None
        tmp_12 = tmp_11.transpose(1, 2);  tmp_11 = None
        tmp_13 = tmp_12.reshape(4, -1, 4);  tmp_12 = None
        tmp_14 = tmp_8.reshape(4, -1, 4)
        tmp_15 = tmp_10.reshape(4, -1, 4)
        tmp_16 = tmp_14.transpose(1, 2);  tmp_14 = None
        bmm = torch.bmm(tmp_13, tmp_16);  tmp_13 = tmp_16 = None
        tmp_18 = bmm.view(1, 4, 22, 22);  bmm = None
        tmp_19 = tmp_18 + in_0;  tmp_18 = in_0 = None
        tmp_20 = tmp_19.view(4, 22, 22);  tmp_19 = None
        tmp_21 = torch.nn.functional.softmax(tmp_20, dim = -1);  tmp_20 = None
        tmp_22 = torch.nn.functional.dropout(tmp_21, p = 0.1, training = False);  tmp_21 = None
        bmm_1 = torch.bmm(tmp_22, tmp_15);  tmp_22 = tmp_15 = None
        tmp_24 = bmm_1.view(1, 4, 22, 4);  bmm_1 = None
        tmp_25 = tmp_24.transpose(1, 2);  tmp_24 = None
        tmp_26 = tmp_25.reshape(1, 22, 16);  tmp_25 = None
        return (tmp_26, tmp_8, tmp_10)
        