import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor):
        tmp_9 = torch.nn.functional.silu(in_9, inplace = True);  in_9 = None
        tmp_10 = tmp_9.mean((2, 3), keepdim = True)
        to = tmp_10.to(torch.bfloat16);  tmp_10 = None
        conv2d = torch.conv2d(to, in_2, in_1, (1, 1), (0, 0), (1, 1), 1);  to = in_2 = in_1 = None
        tmp_12 = torch.nn.functional.silu(conv2d, inplace = True);  conv2d = None
        to_1 = tmp_12.to(torch.bfloat16);  tmp_12 = None
        conv2d_1 = torch.conv2d(to_1, in_4, in_3, (1, 1), (0, 0), (1, 1), 1);  to_1 = in_4 = in_3 = None
        tmp_14 = conv2d_1.sigmoid();  conv2d_1 = None
        tmp_15 = tmp_9 * tmp_14;  tmp_9 = tmp_14 = None
        to_2 = tmp_15.to(torch.bfloat16);  tmp_15 = None
        conv2d_2 = torch.conv2d(to_2, in_0, None, (1, 1), (0, 0), (1, 1), 1);  to_2 = in_0 = None
        tmp_17 = conv2d_2 + in_10;  conv2d_2 = in_10 = None
        tmp_18 = torch.nn.functional.batch_norm(tmp_17, in_5, in_6, in_8, in_7, False, 0.1, 1e-05);  in_5 = in_6 = in_8 = in_7 = None
        tmp_19 = torch.nn.functional.silu(tmp_18, inplace = True);  tmp_18 = None
        return (tmp_17, tmp_19)
        