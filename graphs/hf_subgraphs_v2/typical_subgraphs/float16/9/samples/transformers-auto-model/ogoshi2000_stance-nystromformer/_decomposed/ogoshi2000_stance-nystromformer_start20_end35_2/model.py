import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, in_0, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_0, w_2, w_1);  in_0 = w_2 = w_1 = None
        tmp_4 = linear.view(1, -1, 12, 64);  linear = None
        tmp_5 = tmp_4.transpose(1, 2);  tmp_4 = None
        tmp_6 = in_3 / 2.8284271247461903;  in_3 = None
        tmp_7 = in_2 / 2.8284271247461903;  in_2 = None
        tmp_8 = tmp_7.transpose(-1, -2);  tmp_7 = None
        matmul = torch.matmul(tmp_6, tmp_8);  tmp_6 = tmp_8 = None
        tmp_10 = matmul + in_1;  matmul = in_1 = None
        tmp_11 = torch.nn.functional.softmax(tmp_10, dim = -1);  tmp_10 = None
        matmul_1 = torch.matmul(tmp_11, tmp_5);  tmp_11 = None
        conv2d = torch.conv2d(tmp_5, w_0, None, (1, 1), (32, 0), (1, 1), 12);  tmp_5 = w_0 = None
        matmul_1 += conv2d;  tmp_14 = matmul_1;  matmul_1 = conv2d = None
        tmp_15 = tmp_14.permute(0, 2, 1, 3);  tmp_14 = None
        tmp_16 = tmp_15.contiguous();  tmp_15 = None
        tmp_17 = tmp_16.view(1, 16, 768);  tmp_16 = None
        return (tmp_17,)
        