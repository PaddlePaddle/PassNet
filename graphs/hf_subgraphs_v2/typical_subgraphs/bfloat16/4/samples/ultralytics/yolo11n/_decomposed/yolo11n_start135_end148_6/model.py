import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.view(12, 2, 128, 400);  in_0 = None
        split = tmp_0.split([32, 32, 64], dim = 2);  tmp_0 = None
        tmp_2 = split[0]
        tmp_3 = split[1]
        tmp_4 = split[2];  split = None
        tmp_5 = tmp_2.transpose(-2, -1);  tmp_2 = None
        matmul = tmp_5 @ tmp_3;  tmp_5 = tmp_3 = None
        tmp_7 = matmul * 0.1767766952966369;  matmul = None
        tmp_8 = tmp_7.softmax(dim = -1);  tmp_7 = None
        tmp_9 = tmp_8.transpose(-2, -1);  tmp_8 = None
        matmul_1 = tmp_4 @ tmp_9;  tmp_9 = None
        tmp_11 = matmul_1.view(12, 128, 20, 20);  matmul_1 = None
        tmp_12 = tmp_4.reshape(12, 128, 20, 20);  tmp_4 = None
        return (tmp_12, tmp_11)
        