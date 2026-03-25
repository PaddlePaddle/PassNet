import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, in_3, in_4, in_5, in_6):
        tmp_19 = in_4.transpose(2, 3);  in_4 = None
        matmul = torch.matmul(in_5, tmp_19);  in_5 = tmp_19 = None
        tmp_21 = matmul * 1.0;  matmul = None
        tmp_22 = torch.nn.functional.softmax(tmp_21, dim = -1, dtype = torch.float32);  tmp_21 = None
        tmp_23 = tmp_22.to(torch.float32);  tmp_22 = None
        tmp_24 = torch.nn.functional.dropout(tmp_23, p = 0.0, training = False);  tmp_23 = None
        to_1 = tmp_24.to(torch.float16);  tmp_24 = None
        matmul_1 = torch.matmul(to_1, in_6);  to_1 = in_6 = None
        tmp_26 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_27 = tmp_26.contiguous();  tmp_26 = None
        tmp_28 = tmp_27.reshape(1, 257, -1);  tmp_27 = None
        tmp_29 = tmp_28.contiguous();  tmp_28 = None
        linear = torch.nn.functional.linear(tmp_29, w_13, w_12);  tmp_29 = w_13 = w_12 = None
        tmp_31 = in_3 + linear;  in_3 = linear = None
        tmp_32 = torch.nn.functional.layer_norm(tmp_31, (1280,), w_7, w_6, 1e-05);  w_7 = w_6 = None
        linear_1 = torch.nn.functional.linear(tmp_32, w_9, w_8);  tmp_32 = w_9 = w_8 = None
        tmp_34 = 1.702 * linear_1
        tmp_35 = torch.sigmoid(tmp_34);  tmp_34 = None
        tmp_36 = linear_1 * tmp_35;  linear_1 = tmp_35 = None
        linear_2 = torch.nn.functional.linear(tmp_36, w_11, w_10);  tmp_36 = w_11 = w_10 = None
        tmp_38 = tmp_31 + linear_2;  tmp_31 = linear_2 = None
        tmp_39 = tmp_38[(slice(None, None, None), 0, slice(None, None, None))]
        tmp_40 = torch.nn.functional.layer_norm(tmp_39, (1280,), w_15, w_14, 1e-05);  tmp_39 = w_15 = w_14 = None
        tmp_41 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_42 = tmp_41.to(dtype = torch.float32);  tmp_41 = None
        tmp_43 = 1.0 - tmp_42;  tmp_42 = None
        tmp_44 = tmp_43 * -3.4028234663852886e+38;  tmp_43 = None
        tmp_45 = w_0[(slice(None, None, None), slice(None, 7, None))];  w_0 = None
        tmp_46 = torch.nn.functional.embedding(in_1, w_5, 0, None, 2.0, False, False);  in_1 = w_5 = None
        tmp_47 = torch.nn.functional.embedding(in_2, w_4, None, None, 2.0, False, False);  in_2 = w_4 = None
        tmp_48 = tmp_46 + tmp_47;  tmp_46 = tmp_47 = None
        tmp_49 = torch.nn.functional.embedding(tmp_45, w_3, None, None, 2.0, False, False);  tmp_45 = w_3 = None
        tmp_48 += tmp_49;  tmp_50 = tmp_48;  tmp_48 = tmp_49 = None
        tmp_51 = torch.nn.functional.layer_norm(tmp_50, (1024,), w_2, w_1, 1e-12);  tmp_50 = w_2 = w_1 = None
        tmp_52 = torch.nn.functional.dropout(tmp_51, 0.1, False, False);  tmp_51 = None
        return (tmp_52, tmp_44, tmp_38, tmp_40)
        