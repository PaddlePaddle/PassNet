import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1, in_2, in_3):
        tmp_10 = in_3[(slice(None, None, None), slice(None, None, None), slice(None, 2048, None))]
        tmp_11 = in_3[(slice(None, None, None), slice(None, None, None), slice(2048, None, None))];  in_3 = None
        tmp_12 = torch.nn.functional.gelu(tmp_10, approximate = 'none');  tmp_10 = None
        tmp_13 = tmp_12 * tmp_11;  tmp_12 = tmp_11 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.1, False, False);  tmp_13 = None
        linear = torch.nn.functional.linear(tmp_14, w_3, w_2);  tmp_14 = w_3 = w_2 = None
        tmp_16 = linear + in_2;  linear = in_2 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (512,), w_1, w_0, 1e-12);  tmp_16 = w_1 = w_0 = None
        linear_1 = torch.nn.functional.linear(tmp_17, w_7, w_6);  w_7 = w_6 = None
        linear_2 = torch.nn.functional.linear(tmp_17, w_5, w_4);  w_5 = w_4 = None
        tmp_20 = linear_2.view((1, 11, 8, 64));  linear_2 = None
        tmp_21 = tmp_20.permute(0, 2, 1, 3);  tmp_20 = None
        linear_3 = torch.nn.functional.linear(tmp_17, w_9, w_8);  w_9 = w_8 = None
        tmp_23 = linear_3.view((1, 11, 8, 64));  linear_3 = None
        tmp_24 = tmp_23.permute(0, 2, 1, 3);  tmp_23 = None
        tmp_25 = linear_1.view((1, 11, 8, 64));  linear_1 = None
        tmp_26 = tmp_25.permute(0, 2, 1, 3);  tmp_25 = None
        tmp_27 = tmp_21.transpose(-1, -2);  tmp_21 = None
        matmul = torch.matmul(tmp_26, tmp_27);  tmp_26 = tmp_27 = None
        tmp_29 = matmul / 8.0;  matmul = None
        tmp_30 = tmp_29 + in_1;  tmp_29 = in_1 = None
        tmp_31 = tmp_30 + in_0;  tmp_30 = in_0 = None
        tmp_32 = torch.nn.functional.softmax(tmp_31, dim = -1);  tmp_31 = None
        tmp_33 = torch.nn.functional.dropout(tmp_32, 0.0, False, False);  tmp_32 = None
        matmul_1 = torch.matmul(tmp_33, tmp_24);  tmp_33 = tmp_24 = None
        tmp_35 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_36 = tmp_35.contiguous();  tmp_35 = None
        tmp_37 = tmp_36.view((1, 11, 512));  tmp_36 = None
        return (tmp_37, tmp_17)
        