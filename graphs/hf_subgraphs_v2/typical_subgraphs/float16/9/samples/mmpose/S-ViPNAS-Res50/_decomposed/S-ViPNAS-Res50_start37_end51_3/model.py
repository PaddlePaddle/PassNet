import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0, in_1):
        tmp_8 = in_1.view(1, 80, 3072)
        tmp_9 = tmp_8.unsqueeze(1);  tmp_8 = None
        conv2d = torch.conv2d(in_1, w_7, w_6, (1, 1), (0, 0), (1, 1), 1);  w_7 = w_6 = None
        tmp_11 = conv2d.view(1, 1, 3072);  conv2d = None
        tmp_12 = torch.nn.functional.softmax(tmp_11, 2, _stacklevel = 5);  tmp_11 = None
        tmp_13 = tmp_12.unsqueeze(-1);  tmp_12 = None
        matmul = torch.matmul(tmp_9, tmp_13);  tmp_9 = tmp_13 = None
        tmp_15 = matmul.view(1, 80, 1, 1);  matmul = None
        conv2d_1 = torch.conv2d(tmp_15, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_15 = w_1 = w_0 = None
        tmp_17 = torch.nn.functional.layer_norm(conv2d_1, (16, 1, 1), w_3, w_2, 1e-05);  conv2d_1 = w_3 = w_2 = None
        tmp_18 = torch.nn.functional.relu(tmp_17, inplace = True);  tmp_17 = None
        conv2d_2 = torch.conv2d(tmp_18, w_5, w_4, (1, 1), (0, 0), (1, 1), 1);  tmp_18 = w_5 = w_4 = None
        tmp_20 = in_1 + conv2d_2;  in_1 = conv2d_2 = None
        tmp_20 += in_0;  tmp_21 = tmp_20;  tmp_20 = in_0 = None
        return (tmp_21,)
        