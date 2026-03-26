import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.flatten(2);  in_0 = None
        tmp_1 = tmp_0.transpose(1, 2);  tmp_0 = None
        tmp_2 = tmp_1.view(1, 400, 12, 96);  tmp_1 = None
        tmp_3 = tmp_2.permute(0, 2, 3, 1);  tmp_2 = None
        split = tmp_3.split([32, 32, 32], dim = 2);  tmp_3 = None
        tmp_5 = split[0]
        tmp_6 = split[1]
        tmp_7 = split[2];  split = None
        tmp_8 = tmp_5.transpose(-2, -1);  tmp_5 = None
        matmul = tmp_8 @ tmp_6;  tmp_8 = tmp_6 = None
        tmp_10 = matmul * 0.1767766952966369;  matmul = None
        tmp_11 = tmp_10.softmax(dim = -1);  tmp_10 = None
        tmp_12 = tmp_11.transpose(-2, -1);  tmp_11 = None
        matmul_1 = tmp_7 @ tmp_12;  tmp_12 = None
        tmp_14 = matmul_1.permute(0, 3, 1, 2);  matmul_1 = None
        tmp_15 = tmp_7.permute(0, 3, 1, 2);  tmp_7 = None
        tmp_16 = tmp_14.reshape(1, 20, 20, 384);  tmp_14 = None
        tmp_17 = tmp_16.permute(0, 3, 1, 2);  tmp_16 = None
        tmp_18 = tmp_17.contiguous();  tmp_17 = None
        tmp_19 = tmp_15.reshape(1, 20, 20, 384);  tmp_15 = None
        tmp_20 = tmp_19.permute(0, 3, 1, 2);  tmp_19 = None
        tmp_21 = tmp_20.contiguous();  tmp_20 = None
        return (tmp_21, tmp_18)
        