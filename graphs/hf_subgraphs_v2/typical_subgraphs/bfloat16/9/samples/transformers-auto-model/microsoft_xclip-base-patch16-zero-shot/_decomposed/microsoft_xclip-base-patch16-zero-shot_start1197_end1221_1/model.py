import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, in_0, in_1, in_2):
        tmp_7 = 1.702 * in_0
        tmp_8 = torch.sigmoid(tmp_7);  tmp_7 = None
        tmp_9 = in_0 * tmp_8;  in_0 = tmp_8 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.0, False, False);  tmp_9 = None
        linear = torch.nn.functional.linear(tmp_10, w_1, w_0);  tmp_10 = w_1 = w_0 = None
        tmp_12 = in_2 + linear;  in_2 = linear = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (512,), w_6, w_5, 1e-05);  w_6 = w_5 = None
        linear_1 = torch.nn.functional.linear(tmp_13, w_3, None);  tmp_13 = w_3 = None
        tmp_15 = linear_1.reshape(1, 1, 8, 64);  linear_1 = None
        tmp_16 = tmp_15.permute(0, 2, 1, 3);  tmp_15 = None
        linear_2 = torch.nn.functional.linear(in_1, w_2, None);  w_2 = None
        tmp_18 = linear_2.reshape(1, 196, 8, 64);  linear_2 = None
        tmp_19 = tmp_18.permute(0, 2, 1, 3);  tmp_18 = None
        linear_3 = torch.nn.functional.linear(in_1, w_4, None);  in_1 = w_4 = None
        tmp_21 = linear_3.reshape(1, 196, 8, 64);  linear_3 = None
        tmp_22 = tmp_21.permute(0, 2, 1, 3);  tmp_21 = None
        tmp_23 = tmp_19.transpose(-2, -1);  tmp_19 = None
        matmul = tmp_16 @ tmp_23;  tmp_16 = tmp_23 = None
        tmp_25 = matmul * 0.125;  matmul = None
        tmp_26 = tmp_25.softmax(dim = -1);  tmp_25 = None
        tmp_27 = torch.nn.functional.dropout(tmp_26, 0.0, False, False);  tmp_26 = None
        matmul_1 = tmp_27 @ tmp_22;  tmp_27 = tmp_22 = None
        tmp_29 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_30 = tmp_29.reshape(1, 1, 512);  tmp_29 = None
        return (tmp_12, tmp_30)
        