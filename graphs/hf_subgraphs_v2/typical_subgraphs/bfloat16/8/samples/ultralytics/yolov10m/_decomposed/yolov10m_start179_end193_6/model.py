import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_1 = in_1.view(64, 4, 144, 400);  in_1 = None
        split = tmp_1.split([36, 36, 72], dim = 2);  tmp_1 = None
        tmp_3 = split[0]
        tmp_4 = split[1]
        tmp_5 = split[2];  split = None
        tmp_6 = tmp_3.transpose(-2, -1);  tmp_3 = None
        matmul = tmp_6 @ tmp_4;  tmp_6 = tmp_4 = None
        item = in_0.item();  in_0 = None
        tmp_9 = matmul * item;  matmul = item = None
        tmp_10 = tmp_9.softmax(dim = -1);  tmp_9 = None
        tmp_11 = tmp_10.transpose(-2, -1);  tmp_10 = None
        matmul_1 = tmp_5 @ tmp_11;  tmp_11 = None
        tmp_13 = matmul_1.view(64, 288, 20, 20);  matmul_1 = None
        tmp_14 = tmp_5.reshape(64, 288, 20, 20);  tmp_5 = None
        return (tmp_14, tmp_13)
        