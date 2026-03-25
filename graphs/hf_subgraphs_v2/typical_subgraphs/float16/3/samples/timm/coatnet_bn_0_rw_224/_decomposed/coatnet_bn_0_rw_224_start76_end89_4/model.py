import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
        tmp_10 = torch.nn.functional.silu(in_11, inplace = True);  in_11 = None
        to = tmp_10.to(torch.float16);  tmp_10 = None
        conv2d = torch.conv2d(to, in_0, None, (1, 1), (1, 1), (1, 1), 768);  to = in_0 = None
        tmp_12 = conv2d.mean((2, 3), keepdim = True)
        to_1 = tmp_12.to(torch.float16);  tmp_12 = None
        conv2d_1 = torch.conv2d(to_1, in_7, in_6, (1, 1), (0, 0), (1, 1), 1);  to_1 = in_7 = in_6 = None
        tmp_14 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        to_2 = tmp_14.to(torch.float16);  tmp_14 = None
        conv2d_2 = torch.conv2d(to_2, in_9, in_8, (1, 1), (0, 0), (1, 1), 1);  to_2 = in_9 = in_8 = None
        tmp_16 = conv2d_2.sigmoid();  conv2d_2 = None
        tmp_17 = conv2d * tmp_16;  conv2d = tmp_16 = None
        tmp_18 = torch.nn.functional.batch_norm(tmp_17, in_2, in_3, in_5, in_4, False, 0.1, 1e-05);  tmp_17 = in_2 = in_3 = in_5 = in_4 = None
        tmp_19 = torch.nn.functional.silu(tmp_18, inplace = True);  tmp_18 = None
        to_3 = tmp_19.to(torch.float16);  tmp_19 = None
        conv2d_3 = torch.conv2d(to_3, in_1, None, (1, 1), (0, 0), (1, 1), 1);  to_3 = in_1 = None
        tmp_21 = conv2d_3 + in_10;  conv2d_3 = in_10 = None
        tmp_22 = torch.nn.functional.avg_pool2d(tmp_21, 2, 2, 0, False, True, None)
        return (tmp_21, tmp_22)
        