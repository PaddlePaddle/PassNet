import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor):
        tmp_7 = torch.nn.functional.relu(in_7, inplace = True);  in_7 = None
        conv2d = torch.conv2d(tmp_7, in_4, in_3, (1, 1), (0, 0), (1, 1), 1);  in_4 = in_3 = None
        tmp_9 = conv2d.reshape(1, 64, -1);  conv2d = None
        tmp_10 = tmp_9.permute(0, 2, 1);  tmp_9 = None
        tmp_11 = tmp_10.contiguous();  tmp_10 = None
        conv2d_1 = torch.conv2d(tmp_7, in_2, in_1, (1, 1), (0, 0), (1, 1), 1);  in_2 = in_1 = None
        conv2d_2 = torch.conv2d(tmp_7, in_6, in_5, (1, 1), (0, 0), (1, 1), 1);  in_6 = in_5 = None
        tmp_14 = conv2d_1.reshape(1, 64, -1);  conv2d_1 = None
        tmp_15 = conv2d_2.reshape(1, 512, -1);  conv2d_2 = None
        tmp_16 = tmp_15.permute(0, 2, 1);  tmp_15 = None
        tmp_17 = tmp_16.contiguous();  tmp_16 = None
        matmul = torch.matmul(tmp_11, tmp_14);  tmp_11 = tmp_14 = None
        tmp_19 = torch.nn.functional.softmax(matmul, dim = -1);  matmul = None
        matmul_1 = torch.matmul(tmp_19, tmp_17);  tmp_19 = tmp_17 = None
        tmp_21 = matmul_1.permute(0, 2, 1);  matmul_1 = None
        tmp_22 = tmp_21.contiguous();  tmp_21 = None
        tmp_23 = tmp_22.reshape(1, -1, 64, 64);  tmp_22 = None
        tmp_24 = tmp_23 * in_0;  tmp_23 = in_0 = None
        tmp_25 = tmp_24 + tmp_7;  tmp_24 = tmp_7 = None
        return (tmp_25,)
        