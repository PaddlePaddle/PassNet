import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22):
        tmp_19 = in_20.transpose(2, 3);  in_20 = None
        matmul = torch.matmul(in_21, tmp_19);  in_21 = tmp_19 = None
        tmp_21 = matmul * 1.0;  matmul = None
        tmp_22 = torch.nn.functional.softmax(tmp_21, dim = -1, dtype = torch.float32);  tmp_21 = None
        tmp_23 = tmp_22.to(torch.float32);  tmp_22 = None
        tmp_24 = torch.nn.functional.dropout(tmp_23, p = 0.0, training = False);  tmp_23 = None
        to_1 = tmp_24.to(torch.float16);  tmp_24 = None
        matmul_1 = torch.matmul(to_1, in_22);  to_1 = in_22 = None
        tmp_26 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_27 = tmp_26.contiguous();  tmp_26 = None
        tmp_28 = tmp_27.reshape(1, 257, -1);  tmp_27 = None
        tmp_29 = tmp_28.contiguous();  tmp_28 = None
        linear = torch.nn.functional.linear(tmp_29, in_16, in_15);  tmp_29 = in_16 = in_15 = None
        tmp_31 = in_19 + linear;  in_19 = linear = None
        tmp_32 = torch.nn.functional.layer_norm(tmp_31, (1024,), in_10, in_9, 1e-05);  in_10 = in_9 = None
        linear_1 = torch.nn.functional.linear(tmp_32, in_12, in_11);  tmp_32 = in_12 = in_11 = None
        tmp_34 = 1.702 * linear_1
        tmp_35 = torch.sigmoid(tmp_34);  tmp_34 = None
        tmp_36 = linear_1 * tmp_35;  linear_1 = tmp_35 = None
        linear_2 = torch.nn.functional.linear(tmp_36, in_14, in_13);  tmp_36 = in_14 = in_13 = None
        tmp_38 = tmp_31 + linear_2;  tmp_31 = linear_2 = None
        tmp_39 = tmp_38[(slice(None, None, None), 0, slice(None, None, None))]
        tmp_40 = torch.nn.functional.layer_norm(tmp_39, (1024,), in_18, in_17, 1e-05);  tmp_39 = in_18 = in_17 = None
        tmp_41 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_42 = tmp_41.to(dtype = torch.float32);  tmp_41 = None
        tmp_43 = 1.0 - tmp_42;  tmp_42 = None
        tmp_44 = tmp_43 * -3.4028234663852886e+38;  tmp_43 = None
        tmp_45 = in_3[(slice(None, None, None), slice(None, 7, None))];  in_3 = None
        tmp_46 = torch.nn.functional.embedding(in_1, in_8, 0, None, 2.0, False, False);  in_1 = in_8 = None
        tmp_47 = torch.nn.functional.embedding(in_2, in_7, None, None, 2.0, False, False);  in_2 = in_7 = None
        tmp_48 = tmp_46 + tmp_47;  tmp_46 = tmp_47 = None
        tmp_49 = torch.nn.functional.embedding(tmp_45, in_6, None, None, 2.0, False, False);  tmp_45 = in_6 = None
        tmp_48 += tmp_49;  tmp_50 = tmp_48;  tmp_48 = tmp_49 = None
        tmp_51 = torch.nn.functional.layer_norm(tmp_50, (768,), in_5, in_4, 1e-12);  tmp_50 = in_5 = in_4 = None
        tmp_52 = torch.nn.functional.dropout(tmp_51, 0.1, False, False);  tmp_51 = None
        return (tmp_52, tmp_44, tmp_38, tmp_40)
        