import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
        tmp_8 = torch.nn.functional.adaptive_avg_pool2d(in_10, 1)
        conv2d = torch.conv2d(tmp_8, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_8 = in_1 = in_0 = None
        tmp_10 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_10, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  tmp_10 = in_3 = in_2 = None
        tmp_12 = torch.sigmoid(conv2d_1);  conv2d_1 = None
        tmp_13 = in_10 * tmp_12;  in_10 = tmp_12 = None
        tmp_14 = torch.nn.functional.adaptive_avg_pool2d(in_11, 1)
        conv2d_2 = torch.conv2d(tmp_14, in_5, in_4, (1, 1), (0, 0), (1, 1), 1);  tmp_14 = in_5 = in_4 = None
        tmp_16 = torch.nn.functional.relu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_16, in_7, in_6, (1, 1), (0, 0), (1, 1), 1);  tmp_16 = in_7 = in_6 = None
        tmp_18 = torch.sigmoid(conv2d_3);  conv2d_3 = None
        tmp_19 = in_11 * tmp_18;  in_11 = tmp_18 = None
        tmp_20 = torch.cat([in_8, tmp_13], dim = 1);  in_8 = tmp_13 = None
        tmp_21 = torch.cat([in_9, tmp_19], dim = 1);  in_9 = tmp_19 = None
        tmp_22 = tmp_20.view(512, 2, 20, 64, 48);  tmp_20 = None
        tmp_23 = torch.transpose(tmp_22, 1, 2);  tmp_22 = None
        tmp_24 = tmp_23.contiguous();  tmp_23 = None
        tmp_25 = tmp_24.view(512, 40, 64, 48);  tmp_24 = None
        tmp_26 = tmp_21.view(512, 2, 40, 32, 24);  tmp_21 = None
        tmp_27 = torch.transpose(tmp_26, 1, 2);  tmp_26 = None
        tmp_28 = tmp_27.contiguous();  tmp_27 = None
        tmp_29 = tmp_28.view(512, 80, 32, 24);  tmp_28 = None
        chunk = tmp_25.chunk(2, dim = 1);  tmp_25 = None
        tmp_31 = chunk[0]
        tmp_32 = chunk[1];  chunk = None
        chunk_1 = tmp_29.chunk(2, dim = 1);  tmp_29 = None
        tmp_34 = chunk_1[0]
        tmp_35 = chunk_1[1];  chunk_1 = None
        tmp_36 = torch.nn.functional.adaptive_avg_pool2d(tmp_32, (32, 24))
        tmp_37 = torch.cat([tmp_36, tmp_35], dim = 1);  tmp_36 = None
        return (tmp_37, tmp_31, tmp_34, tmp_32, tmp_35)
        