import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
        tmp_10 = torch.nn.functional.silu(in_11, inplace = True);  in_11 = None
        conv2d = torch.conv2d(tmp_10, in_0, None, (1, 1), (1, 1), (1, 1), 384);  tmp_10 = in_0 = None
        tmp_12 = conv2d.mean((2, 3), keepdim = True)
        conv2d_1 = torch.conv2d(tmp_12, in_7, in_6, (1, 1), (0, 0), (1, 1), 1);  tmp_12 = in_7 = in_6 = None
        tmp_14 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        conv2d_2 = torch.conv2d(tmp_14, in_9, in_8, (1, 1), (0, 0), (1, 1), 1);  tmp_14 = in_9 = in_8 = None
        tmp_16 = conv2d_2.sigmoid();  conv2d_2 = None
        tmp_17 = conv2d * tmp_16;  conv2d = tmp_16 = None
        tmp_18 = torch.nn.functional.batch_norm(tmp_17, in_2, in_3, in_5, in_4, False, 0.1, 1e-05);  tmp_17 = in_2 = in_3 = in_5 = in_4 = None
        tmp_19 = torch.nn.functional.silu(tmp_18, inplace = True);  tmp_18 = None
        conv2d_3 = torch.conv2d(tmp_19, in_1, None, (1, 1), (0, 0), (1, 1), 1);  tmp_19 = in_1 = None
        tmp_21 = conv2d_3 + in_10;  conv2d_3 = in_10 = None
        tmp_22 = torch.nn.functional.max_pool2d(tmp_21, 3, 2, 1, 1, ceil_mode = False, return_indices = False)
        return (tmp_21, tmp_22)
        