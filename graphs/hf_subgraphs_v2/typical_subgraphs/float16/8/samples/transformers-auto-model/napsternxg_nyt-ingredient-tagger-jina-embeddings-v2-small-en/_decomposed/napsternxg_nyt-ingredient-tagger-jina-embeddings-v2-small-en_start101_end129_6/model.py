import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13):
        tmp_10 = in_13[(slice(None, None, None), slice(None, None, None), slice(None, 2048, None))]
        tmp_11 = in_13[(slice(None, None, None), slice(None, None, None), slice(2048, None, None))];  in_13 = None
        tmp_12 = torch.nn.functional.gelu(tmp_10, approximate = 'none');  tmp_10 = None
        tmp_13 = tmp_12 * tmp_11;  tmp_12 = tmp_11 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.1, False, False);  tmp_13 = None
        linear = torch.nn.functional.linear(tmp_14, in_3, in_2);  tmp_14 = in_3 = in_2 = None
        tmp_16 = linear + in_12;  linear = in_12 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (512,), in_1, in_0, 1e-12);  tmp_16 = in_1 = in_0 = None
        linear_1 = torch.nn.functional.linear(tmp_17, in_7, in_6);  in_7 = in_6 = None
        linear_2 = torch.nn.functional.linear(tmp_17, in_5, in_4);  in_5 = in_4 = None
        tmp_20 = linear_2.view((128, 64, 8, 64));  linear_2 = None
        tmp_21 = tmp_20.permute(0, 2, 1, 3);  tmp_20 = None
        linear_3 = torch.nn.functional.linear(tmp_17, in_9, in_8);  in_9 = in_8 = None
        tmp_23 = linear_3.view((128, 64, 8, 64));  linear_3 = None
        tmp_24 = tmp_23.permute(0, 2, 1, 3);  tmp_23 = None
        tmp_25 = linear_1.view((128, 64, 8, 64));  linear_1 = None
        tmp_26 = tmp_25.permute(0, 2, 1, 3);  tmp_25 = None
        tmp_27 = tmp_21.transpose(-1, -2);  tmp_21 = None
        matmul = torch.matmul(tmp_26, tmp_27);  tmp_26 = tmp_27 = None
        tmp_29 = matmul / 8.0;  matmul = None
        tmp_30 = tmp_29 + in_10;  tmp_29 = in_10 = None
        tmp_31 = tmp_30 + in_11;  tmp_30 = in_11 = None
        tmp_32 = torch.nn.functional.softmax(tmp_31, dim = -1);  tmp_31 = None
        tmp_33 = torch.nn.functional.dropout(tmp_32, 0.0, False, False);  tmp_32 = None
        matmul_1 = torch.matmul(tmp_33, tmp_24);  tmp_33 = tmp_24 = None
        tmp_35 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_36 = tmp_35.contiguous();  tmp_35 = None
        tmp_37 = tmp_36.view((128, 64, 512));  tmp_36 = None
        return (tmp_17, tmp_37)
        