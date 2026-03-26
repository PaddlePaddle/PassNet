import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15):
        tmp_12 = in_13.transpose(2, 3);  in_13 = None
        matmul = torch.matmul(in_14, tmp_12);  in_14 = tmp_12 = None
        tmp_14 = matmul * 1.0;  matmul = None
        tmp_15 = torch.nn.functional.softmax(tmp_14, dim = -1, dtype = torch.float32);  tmp_14 = None
        tmp_16 = tmp_15.to(torch.float32);  tmp_15 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, p = 0.0, training = False);  tmp_16 = None
        to_1 = tmp_17.to(torch.float16);  tmp_17 = None
        matmul_1 = torch.matmul(to_1, in_15);  to_1 = in_15 = None
        tmp_19 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_20 = tmp_19.contiguous();  tmp_19 = None
        tmp_21 = tmp_20.reshape(1, 257, -1);  tmp_20 = None
        tmp_22 = tmp_21.contiguous();  tmp_21 = None
        linear = torch.nn.functional.linear(tmp_22, in_11, in_10);  tmp_22 = in_11 = in_10 = None
        tmp_24 = in_12 + linear;  in_12 = linear = None
        tmp_25 = torch.nn.functional.layer_norm(tmp_24, (1280,), in_5, in_4, 1e-05);  in_5 = in_4 = None
        linear_1 = torch.nn.functional.linear(tmp_25, in_7, in_6);  tmp_25 = in_7 = in_6 = None
        tmp_27 = 1.702 * linear_1
        tmp_28 = torch.sigmoid(tmp_27);  tmp_27 = None
        tmp_29 = linear_1 * tmp_28;  linear_1 = tmp_28 = None
        linear_2 = torch.nn.functional.linear(tmp_29, in_9, in_8);  tmp_29 = in_9 = in_8 = None
        tmp_31 = tmp_24 + linear_2;  tmp_24 = linear_2 = None
        tmp_32 = torch.nn.functional.layer_norm(tmp_31, (1280,), in_1, in_0, 1e-05);  in_1 = in_0 = None
        linear_3 = torch.nn.functional.linear(tmp_32, in_3, in_2);  in_3 = in_2 = None
        tmp_34 = linear_3.view((1, 257, -1, 80));  linear_3 = None
        tmp_35 = tmp_34.transpose(1, 2);  tmp_34 = None
        tmp_36 = tmp_35 * 0.11180339887498948;  tmp_35 = None
        return (tmp_31, tmp_32, tmp_36)
        