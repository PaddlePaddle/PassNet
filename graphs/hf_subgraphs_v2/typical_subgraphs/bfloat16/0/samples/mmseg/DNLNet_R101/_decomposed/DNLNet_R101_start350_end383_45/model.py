import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor):
        tmp_8 = torch.nn.functional.relu(in_8, inplace = True);  in_8 = None
        conv2d = torch.conv2d(tmp_8, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  in_3 = in_2 = None
        tmp_10 = conv2d.view(1, 256, -1);  conv2d = None
        tmp_11 = tmp_10.permute(0, 2, 1);  tmp_10 = None
        conv2d_1 = torch.conv2d(tmp_8, in_7, in_6, (1, 1), (0, 0), (1, 1), 1);  in_7 = in_6 = None
        tmp_13 = conv2d_1.view(1, 256, -1);  conv2d_1 = None
        tmp_14 = tmp_13.permute(0, 2, 1);  tmp_13 = None
        conv2d_2 = torch.conv2d(tmp_8, in_5, in_4, (1, 1), (0, 0), (1, 1), 1);  in_5 = in_4 = None
        tmp_16 = conv2d_2.view(1, 256, -1);  conv2d_2 = None
        tmp_17 = tmp_14.mean(dim = -2, keepdim = True)
        tmp_14 -= tmp_17;  tmp_18 = tmp_14;  tmp_14 = tmp_17 = None
        tmp_19 = tmp_16.mean(dim = -1, keepdim = True)
        tmp_16 -= tmp_19;  tmp_20 = tmp_16;  tmp_16 = tmp_19 = None
        matmul = torch.matmul(tmp_18, tmp_20);  tmp_18 = tmp_20 = None
        tmp_22 = torch.tensor(256, dtype = torch.float32, device = device(type='cuda', index=0))
        tmp_23 = torch.tensor(0.5, device = device(type='cuda', index=0))
        tmp_24 = tmp_22 ** tmp_23;  tmp_22 = tmp_23 = None
        matmul /= tmp_24;  tmp_25 = matmul;  matmul = tmp_24 = None
        tmp_26 = torch.tensor(0.05, device = device(type='cuda', index=0))
        tmp_25 /= tmp_26;  tmp_27 = tmp_25;  tmp_25 = tmp_26 = None
        tmp_28 = tmp_27.softmax(dim = -1);  tmp_27 = None
        matmul_1 = torch.matmul(tmp_28, tmp_11);  tmp_28 = None
        tmp_30 = matmul_1.permute(0, 2, 1);  matmul_1 = None
        tmp_31 = tmp_30.contiguous();  tmp_30 = None
        tmp_32 = tmp_31.reshape(1, 256, 64, 64);  tmp_31 = None
        conv2d_3 = torch.conv2d(tmp_8, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_1 = in_0 = None
        tmp_34 = conv2d_3.view(1, 1, -1);  conv2d_3 = None
        tmp_35 = tmp_34.softmax(dim = -1);  tmp_34 = None
        matmul_2 = torch.matmul(tmp_35, tmp_11);  tmp_35 = tmp_11 = None
        tmp_37 = matmul_2.permute(0, 2, 1);  matmul_2 = None
        tmp_38 = tmp_37.contiguous();  tmp_37 = None
        tmp_39 = tmp_38.reshape(1, 256, 1, 1);  tmp_38 = None
        tmp_40 = tmp_32 + tmp_39;  tmp_32 = tmp_39 = None
        return (tmp_40, tmp_8)
        