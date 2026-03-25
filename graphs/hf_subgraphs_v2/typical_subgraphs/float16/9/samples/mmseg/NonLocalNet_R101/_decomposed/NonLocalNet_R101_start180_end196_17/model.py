import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, in_0 : torch.Tensor):
        tmp_6 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        conv2d = torch.conv2d(tmp_6, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  w_1 = w_0 = None
        tmp_8 = conv2d.view(1, 256, -1);  conv2d = None
        tmp_9 = tmp_8.permute(0, 2, 1);  tmp_8 = None
        conv2d_1 = torch.conv2d(tmp_6, w_5, w_4, (1, 1), (0, 0), (1, 1), 1);  w_5 = w_4 = None
        tmp_11 = conv2d_1.view(1, 256, -1);  conv2d_1 = None
        tmp_12 = tmp_11.permute(0, 2, 1);  tmp_11 = None
        conv2d_2 = torch.conv2d(tmp_6, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  w_3 = w_2 = None
        tmp_14 = conv2d_2.view(1, 256, -1);  conv2d_2 = None
        matmul = torch.matmul(tmp_12, tmp_14);  tmp_12 = tmp_14 = None
        matmul /= 16.0;  tmp_16 = matmul;  matmul = None
        tmp_17 = tmp_16.softmax(dim = -1);  tmp_16 = None
        matmul_1 = torch.matmul(tmp_17, tmp_9);  tmp_17 = tmp_9 = None
        tmp_19 = matmul_1.permute(0, 2, 1);  matmul_1 = None
        tmp_20 = tmp_19.contiguous();  tmp_19 = None
        tmp_21 = tmp_20.reshape(1, 256, 64, 64);  tmp_20 = None
        return (tmp_6, tmp_21)
        