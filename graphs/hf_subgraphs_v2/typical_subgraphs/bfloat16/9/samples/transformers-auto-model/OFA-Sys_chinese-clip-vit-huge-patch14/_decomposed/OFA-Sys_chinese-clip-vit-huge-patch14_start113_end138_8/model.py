import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0, in_1, in_2, in_3):
        tmp_12 = in_1.transpose(2, 3);  in_1 = None
        matmul = torch.matmul(in_2, tmp_12);  in_2 = tmp_12 = None
        tmp_14 = matmul * 1.0;  matmul = None
        tmp_15 = torch.nn.functional.softmax(tmp_14, dim = -1, dtype = torch.float32);  tmp_14 = None
        tmp_16 = tmp_15.to(torch.float32);  tmp_15 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, p = 0.0, training = False);  tmp_16 = None
        to_1 = tmp_17.to(torch.bfloat16);  tmp_17 = None
        matmul_1 = torch.matmul(to_1, in_3);  to_1 = in_3 = None
        tmp_19 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_20 = tmp_19.contiguous();  tmp_19 = None
        tmp_21 = tmp_20.reshape(1, 257, -1);  tmp_20 = None
        tmp_22 = tmp_21.contiguous();  tmp_21 = None
        linear = torch.nn.functional.linear(tmp_22, w_7, w_6);  tmp_22 = w_7 = w_6 = None
        tmp_24 = in_0 + linear;  in_0 = linear = None
        tmp_25 = torch.nn.functional.layer_norm(tmp_24, (1280,), w_1, w_0, 1e-05);  w_1 = w_0 = None
        linear_1 = torch.nn.functional.linear(tmp_25, w_3, w_2);  tmp_25 = w_3 = w_2 = None
        tmp_27 = 1.702 * linear_1
        tmp_28 = torch.sigmoid(tmp_27);  tmp_27 = None
        tmp_29 = linear_1 * tmp_28;  linear_1 = tmp_28 = None
        linear_2 = torch.nn.functional.linear(tmp_29, w_5, w_4);  tmp_29 = w_5 = w_4 = None
        tmp_31 = tmp_24 + linear_2;  tmp_24 = linear_2 = None
        tmp_32 = torch.nn.functional.layer_norm(tmp_31, (1280,), w_9, w_8, 1e-05);  w_9 = w_8 = None
        linear_3 = torch.nn.functional.linear(tmp_32, w_11, w_10);  w_11 = w_10 = None
        tmp_34 = linear_3.view((1, 257, -1, 80));  linear_3 = None
        tmp_35 = tmp_34.transpose(1, 2);  tmp_34 = None
        tmp_36 = tmp_35 * 0.11180339887498948;  tmp_35 = None
        return (tmp_31, tmp_32, tmp_36)
        