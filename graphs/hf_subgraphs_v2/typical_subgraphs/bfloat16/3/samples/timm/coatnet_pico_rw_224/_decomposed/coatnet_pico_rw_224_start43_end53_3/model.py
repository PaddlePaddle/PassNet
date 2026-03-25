import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
        tmp_10 = torch.nn.functional.silu(in_11, inplace = True);  in_11 = None
        tmp_11 = tmp_10.mean((2, 3), keepdim = True)
        to = tmp_11.to(torch.bfloat16);  tmp_11 = None
        conv2d = torch.conv2d(to, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  to = in_3 = in_2 = None
        tmp_13 = torch.nn.functional.silu(conv2d, inplace = True);  conv2d = None
        to_1 = tmp_13.to(torch.bfloat16);  tmp_13 = None
        conv2d_1 = torch.conv2d(to_1, in_5, in_4, (1, 1), (0, 0), (1, 1), 1);  to_1 = in_5 = in_4 = None
        tmp_15 = conv2d_1.sigmoid();  conv2d_1 = None
        tmp_16 = tmp_10 * tmp_15;  tmp_10 = tmp_15 = None
        to_2 = tmp_16.to(torch.bfloat16);  tmp_16 = None
        conv2d_2 = torch.conv2d(to_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  to_2 = in_1 = in_0 = None
        tmp_18 = conv2d_2 + in_10;  conv2d_2 = in_10 = None
        tmp_19 = torch.nn.functional.batch_norm(tmp_18, in_6, in_7, in_9, in_8, False, 0.1, 1e-05);  in_6 = in_7 = in_9 = in_8 = None
        return (tmp_18, tmp_19)
        