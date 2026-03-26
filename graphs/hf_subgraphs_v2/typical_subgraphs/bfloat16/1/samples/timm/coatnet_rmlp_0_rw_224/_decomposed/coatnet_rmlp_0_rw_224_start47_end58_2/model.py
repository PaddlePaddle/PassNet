import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
        tmp_9 = torch.nn.functional.silu(in_10, inplace = True);  in_10 = None
        tmp_10 = tmp_9.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_10, in_2, in_1, (1, 1), (0, 0), (1, 1), 1);  tmp_10 = in_2 = in_1 = None
        tmp_12 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_12, in_4, in_3, (1, 1), (0, 0), (1, 1), 1);  tmp_12 = in_4 = in_3 = None
        tmp_14 = conv2d_1.sigmoid();  conv2d_1 = None
        tmp_15 = tmp_9 * tmp_14;  tmp_9 = tmp_14 = None
        conv2d_2 = torch.conv2d(tmp_15, in_0, None, (1, 1), (0, 0), (1, 1), 1);  tmp_15 = in_0 = None
        tmp_17 = conv2d_2 + in_9;  conv2d_2 = in_9 = None
        tmp_18 = torch.nn.functional.batch_norm(tmp_17, in_5, in_6, in_8, in_7, False, 0.1, 1e-05);  in_5 = in_6 = in_8 = in_7 = None
        tmp_19 = torch.nn.functional.silu(tmp_18, inplace = True);  tmp_18 = None
        return (tmp_17, tmp_19)
        