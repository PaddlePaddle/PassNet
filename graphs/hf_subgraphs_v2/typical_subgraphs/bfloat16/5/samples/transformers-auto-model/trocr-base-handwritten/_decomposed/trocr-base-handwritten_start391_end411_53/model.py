import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        tmp_4 = in_5 * 0.125;  in_5 = None
        linear = torch.nn.functional.linear(in_4, in_1, in_0);  in_1 = in_0 = None
        linear_1 = torch.nn.functional.linear(in_4, in_3, in_2);  in_4 = in_3 = in_2 = None
        tmp_7 = linear.view(1, -1, 16, 64);  linear = None
        tmp_8 = tmp_7.transpose(1, 2);  tmp_7 = None
        tmp_9 = linear_1.view(1, -1, 16, 64);  linear_1 = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        tmp_11 = tmp_4.view(1, 1, 16, 64);  tmp_4 = None
        tmp_12 = tmp_11.transpose(1, 2);  tmp_11 = None
        tmp_13 = tmp_12.reshape(16, -1, 64);  tmp_12 = None
        tmp_14 = tmp_8.reshape(16, -1, 64);  tmp_8 = None
        tmp_15 = tmp_10.reshape(16, -1, 64);  tmp_10 = None
        tmp_16 = tmp_14.transpose(1, 2);  tmp_14 = None
        to = tmp_13.to(torch.bfloat16);  tmp_13 = None
        to_1 = tmp_16.to(torch.bfloat16);  tmp_16 = None
        bmm = torch.bmm(to, to_1);  to = to_1 = None
        tmp_18 = torch.nn.functional.softmax(bmm, dim = -1);  bmm = None
        tmp_19 = torch.nn.functional.dropout(tmp_18, p = 0.0, training = False);  tmp_18 = None
        to_2 = tmp_19.to(torch.bfloat16);  tmp_19 = None
        to_3 = tmp_15.to(torch.bfloat16);  tmp_15 = None
        bmm_1 = torch.bmm(to_2, to_3);  to_2 = to_3 = None
        tmp_21 = bmm_1.view(1, 16, 1, 64);  bmm_1 = None
        tmp_22 = tmp_21.transpose(1, 2);  tmp_21 = None
        tmp_23 = tmp_22.reshape(1, 1, 1024);  tmp_22 = None
        return (tmp_23,)
        