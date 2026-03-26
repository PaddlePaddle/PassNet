import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.flatten(2);  in_0 = None
        tmp_1 = tmp_0.transpose(1, 2);  tmp_0 = None
        tmp_2 = tmp_1.reshape(4, 400, 192);  tmp_1 = None
        tmp_3 = tmp_2.view(4, 400, 2, 96);  tmp_2 = None
        tmp_4 = tmp_3.permute(0, 2, 3, 1);  tmp_3 = None
        split = tmp_4.split([32, 32, 32], dim = 2);  tmp_4 = None
        tmp_6 = split[0]
        tmp_7 = split[1]
        tmp_8 = split[2];  split = None
        tmp_9 = tmp_6.transpose(-2, -1);  tmp_6 = None
        matmul = tmp_9 @ tmp_7;  tmp_9 = tmp_7 = None
        tmp_11 = matmul * 0.1767766952966369;  matmul = None
        tmp_12 = tmp_11.softmax(dim = -1);  tmp_11 = None
        tmp_13 = tmp_12.transpose(-2, -1);  tmp_12 = None
        matmul_1 = tmp_8 @ tmp_13;  tmp_13 = None
        tmp_15 = matmul_1.permute(0, 3, 1, 2);  matmul_1 = None
        tmp_16 = tmp_8.permute(0, 3, 1, 2);  tmp_8 = None
        tmp_17 = tmp_15.reshape(1, 1600, 64);  tmp_15 = None
        tmp_18 = tmp_16.reshape(1, 1600, 64);  tmp_16 = None
        tmp_19 = tmp_17.reshape(1, 40, 40, 64);  tmp_17 = None
        tmp_20 = tmp_19.permute(0, 3, 1, 2);  tmp_19 = None
        tmp_21 = tmp_20.contiguous();  tmp_20 = None
        tmp_22 = tmp_18.reshape(1, 40, 40, 64);  tmp_18 = None
        tmp_23 = tmp_22.permute(0, 3, 1, 2);  tmp_22 = None
        tmp_24 = tmp_23.contiguous();  tmp_23 = None
        return (tmp_24, tmp_21)
        