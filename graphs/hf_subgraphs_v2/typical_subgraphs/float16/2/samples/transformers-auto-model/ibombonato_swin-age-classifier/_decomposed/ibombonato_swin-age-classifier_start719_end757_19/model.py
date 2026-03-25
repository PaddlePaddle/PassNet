import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17):
        linear = torch.nn.functional.linear(in_14, in_4, in_3);  in_14 = in_4 = in_3 = None
        tmp_15 = linear.view((1, 49, -1, 32));  linear = None
        tmp_16 = tmp_15.transpose(1, 2);  tmp_15 = None
        tmp_17 = in_15.transpose(-1, -2);  in_15 = None
        matmul = torch.matmul(in_17, tmp_17);  in_17 = tmp_17 = None
        tmp_19 = matmul / 5.656854249492381;  matmul = None
        tmp_20 = in_2.view(-1);  in_2 = None
        tmp_21 = in_5[tmp_20];  in_5 = tmp_20 = None
        tmp_22 = tmp_21.view(49, 49, -1);  tmp_21 = None
        tmp_23 = tmp_22.permute(2, 0, 1);  tmp_22 = None
        tmp_24 = tmp_23.contiguous();  tmp_23 = None
        tmp_25 = tmp_24.unsqueeze(0);  tmp_24 = None
        tmp_26 = tmp_19 + tmp_25;  tmp_19 = tmp_25 = None
        tmp_27 = torch.nn.functional.softmax(tmp_26, dim = -1);  tmp_26 = None
        tmp_28 = torch.nn.functional.dropout(tmp_27, 0.0, False, False);  tmp_27 = None
        matmul_1 = torch.matmul(tmp_28, tmp_16);  tmp_28 = tmp_16 = None
        tmp_30 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_31 = tmp_30.contiguous();  tmp_30 = None
        tmp_32 = tmp_31.view((1, 49, 768));  tmp_31 = None
        linear_1 = torch.nn.functional.linear(tmp_32, in_1, in_0);  tmp_32 = in_1 = in_0 = None
        tmp_34 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_35 = tmp_34.view(-1, 7, 7, 768);  tmp_34 = None
        tmp_36 = tmp_35.view(-1, 1, 1, 7, 7, 768);  tmp_35 = None
        tmp_37 = tmp_36.permute(0, 1, 3, 2, 4, 5);  tmp_36 = None
        tmp_38 = tmp_37.contiguous();  tmp_37 = None
        tmp_39 = tmp_38.view(-1, 7, 7, 768);  tmp_38 = None
        tmp_40 = tmp_39.view(1, 49, 768);  tmp_39 = None
        tmp_41 = in_16 + tmp_40;  in_16 = tmp_40 = None
        tmp_42 = torch.nn.functional.layer_norm(tmp_41, (768,), in_9, in_8, 1e-05);  in_9 = in_8 = None
        linear_2 = torch.nn.functional.linear(tmp_42, in_7, in_6);  tmp_42 = in_7 = in_6 = None
        tmp_44 = torch.nn.functional.gelu(linear_2);  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_44, in_11, in_10);  tmp_44 = in_11 = in_10 = None
        tmp_46 = torch.nn.functional.dropout(linear_3, 0.0, False, False);  linear_3 = None
        tmp_47 = tmp_41 + tmp_46;  tmp_41 = tmp_46 = None
        tmp_48 = torch.nn.functional.layer_norm(tmp_47, (768,), in_13, in_12, 1e-05);  tmp_47 = in_13 = in_12 = None
        tmp_49 = tmp_48.transpose(1, 2)
        tmp_50 = torch.adaptive_avg_pool1d(tmp_49, 1);  tmp_49 = None
        tmp_51 = torch.flatten(tmp_50, 1);  tmp_50 = None
        return (tmp_48, tmp_51)
        