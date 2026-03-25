import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0, in_1, in_2):
        tmp_8 = in_1 + in_2;  in_1 = in_2 = None
        tmp_8 += in_0;  tmp_9 = tmp_8;  tmp_8 = in_0 = None
        tmp_10 = tmp_9.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_10, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_10 = w_1 = w_0 = None
        tmp_12 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_12, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  tmp_12 = w_3 = w_2 = None
        tmp_14 = conv2d_1.sigmoid();  conv2d_1 = None
        tmp_15 = tmp_9 * tmp_14;  tmp_9 = tmp_14 = None
        tmp_16 = torch.nn.functional.relu(tmp_15, inplace = True);  tmp_15 = None
        tmp_17 = torch.nn.functional.batch_norm(tmp_16, w_4, w_5, w_7, w_6, False, 0.1, 1e-05);  w_4 = w_5 = w_7 = w_6 = None
        return (tmp_16, tmp_17)
        