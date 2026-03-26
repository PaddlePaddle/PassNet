import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1):
        tmp_10 = torch.nn.functional.silu(in_1, inplace = True);  in_1 = None
        conv2d = torch.conv2d(tmp_10, w_0, None, (1, 1), (1, 1), (1, 1), 384);  tmp_10 = w_0 = None
        tmp_12 = conv2d.mean((2, 3), keepdim = True)
        conv2d_1 = torch.conv2d(tmp_12, w_7, w_6, (1, 1), (0, 0), (1, 1), 1);  tmp_12 = w_7 = w_6 = None
        tmp_14 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        conv2d_2 = torch.conv2d(tmp_14, w_9, w_8, (1, 1), (0, 0), (1, 1), 1);  tmp_14 = w_9 = w_8 = None
        tmp_16 = conv2d_2.sigmoid();  conv2d_2 = None
        tmp_17 = conv2d * tmp_16;  conv2d = tmp_16 = None
        tmp_18 = torch.nn.functional.batch_norm(tmp_17, w_2, w_3, w_5, w_4, False, 0.1, 1e-05);  tmp_17 = w_2 = w_3 = w_5 = w_4 = None
        tmp_19 = torch.nn.functional.silu(tmp_18, inplace = True);  tmp_18 = None
        conv2d_3 = torch.conv2d(tmp_19, w_1, None, (1, 1), (0, 0), (1, 1), 1);  tmp_19 = w_1 = None
        tmp_21 = conv2d_3 + in_0;  conv2d_3 = in_0 = None
        tmp_22 = torch.nn.functional.avg_pool2d(tmp_21, 2, 2, 0, False, True, None)
        return (tmp_21, tmp_22)
        