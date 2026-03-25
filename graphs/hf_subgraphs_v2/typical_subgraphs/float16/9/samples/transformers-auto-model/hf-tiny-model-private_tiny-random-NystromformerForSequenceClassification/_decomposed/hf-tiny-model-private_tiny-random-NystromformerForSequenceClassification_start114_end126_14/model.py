import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0, in_1, in_2, in_3):
        tmp_1 = in_2 / 1.6817928305074292;  in_2 = None
        tmp_2 = in_1 / 1.6817928305074292;  in_1 = None
        tmp_3 = tmp_2.transpose(-1, -2);  tmp_2 = None
        matmul = torch.matmul(tmp_1, tmp_3);  tmp_1 = tmp_3 = None
        tmp_5 = matmul + in_0;  matmul = in_0 = None
        tmp_6 = torch.nn.functional.softmax(tmp_5, dim = -1);  tmp_5 = None
        matmul_1 = torch.matmul(tmp_6, in_3);  tmp_6 = None
        conv2d = torch.conv2d(in_3, w_0, None, (1, 1), (32, 0), (1, 1), 4);  in_3 = w_0 = None
        matmul_1 += conv2d;  tmp_9 = matmul_1;  matmul_1 = conv2d = None
        tmp_10 = tmp_9.permute(0, 2, 1, 3);  tmp_9 = None
        tmp_11 = tmp_10.contiguous();  tmp_10 = None
        tmp_12 = tmp_11.view(1, 12, 32);  tmp_11 = None
        return (tmp_12,)
        