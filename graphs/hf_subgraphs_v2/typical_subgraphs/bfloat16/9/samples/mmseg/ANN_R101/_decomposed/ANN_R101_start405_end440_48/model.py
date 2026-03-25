import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, in_0, in_1, in_2):
        tmp_4 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        conv2d = torch.conv2d(in_1, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  w_3 = w_2 = None
        tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
        tmp_7 = tmp_6.view(1, 256, -1);  tmp_6 = None
        tmp_8 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 3)
        tmp_9 = tmp_8.view(1, 256, -1);  tmp_8 = None
        tmp_10 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 6)
        tmp_11 = tmp_10.view(1, 256, -1);  tmp_10 = None
        tmp_12 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 8);  tmp_4 = None
        tmp_13 = tmp_12.view(1, 256, -1);  tmp_12 = None
        tmp_14 = torch.cat([tmp_7, tmp_9, tmp_11, tmp_13], dim = 2);  tmp_7 = tmp_9 = tmp_11 = tmp_13 = None
        tmp_15 = torch.nn.functional.adaptive_avg_pool2d(conv2d, 1)
        tmp_16 = tmp_15.view(1, 256, -1);  tmp_15 = None
        tmp_17 = torch.nn.functional.adaptive_avg_pool2d(conv2d, 3)
        tmp_18 = tmp_17.view(1, 256, -1);  tmp_17 = None
        tmp_19 = torch.nn.functional.adaptive_avg_pool2d(conv2d, 6)
        tmp_20 = tmp_19.view(1, 256, -1);  tmp_19 = None
        tmp_21 = torch.nn.functional.adaptive_avg_pool2d(conv2d, 8);  conv2d = None
        tmp_22 = tmp_21.view(1, 256, -1);  tmp_21 = None
        tmp_23 = torch.cat([tmp_16, tmp_18, tmp_20, tmp_22], dim = 2);  tmp_16 = tmp_18 = tmp_20 = tmp_22 = None
        tmp_24 = tmp_14.reshape(1, 256, -1);  tmp_14 = None
        tmp_25 = tmp_23.reshape(1, 256, -1);  tmp_23 = None
        tmp_26 = tmp_25.permute(0, 2, 1);  tmp_25 = None
        tmp_27 = tmp_26.contiguous();  tmp_26 = None
        matmul = torch.matmul(in_0, tmp_24);  in_0 = tmp_24 = None
        tmp_29 = 0.0625 * matmul;  matmul = None
        tmp_30 = torch.nn.functional.softmax(tmp_29, dim = -1);  tmp_29 = None
        matmul_1 = torch.matmul(tmp_30, tmp_27);  tmp_30 = tmp_27 = None
        tmp_32 = matmul_1.permute(0, 2, 1);  matmul_1 = None
        tmp_33 = tmp_32.contiguous();  tmp_32 = None
        tmp_34 = tmp_33.reshape(1, -1, 64, 128);  tmp_33 = None
        conv2d_1 = torch.conv2d(tmp_34, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_34 = w_1 = w_0 = None
        tmp_36 = torch.stack([conv2d_1], dim = 0);  conv2d_1 = None
        tmp_37 = tmp_36.sum(dim = 0);  tmp_36 = None
        tmp_38 = torch.cat([tmp_37, in_1], 1);  tmp_37 = in_1 = None
        return (tmp_38,)
        