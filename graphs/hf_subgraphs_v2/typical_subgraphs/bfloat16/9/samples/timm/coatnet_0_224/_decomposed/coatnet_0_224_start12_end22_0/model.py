import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_10 = torch.nn.functional.gelu(in_0);  in_0 = None
        tmp_11 = tmp_10.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_11, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  tmp_11 = w_3 = w_2 = None
        tmp_13 = torch.nn.functional.silu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_13, w_5, w_4, (1, 1), (0, 0), (1, 1), 1);  tmp_13 = w_5 = w_4 = None
        tmp_15 = conv2d_1.sigmoid();  conv2d_1 = None
        tmp_16 = tmp_10 * tmp_15;  tmp_10 = tmp_15 = None
        conv2d_2 = torch.conv2d(tmp_16, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_16 = w_1 = w_0 = None
        tmp_18 = conv2d_2 + in_1;  conv2d_2 = in_1 = None
        tmp_19 = torch.nn.functional.batch_norm(tmp_18, w_6, w_7, w_9, w_8, False, 0.1, 1e-05);  w_6 = w_7 = w_9 = w_8 = None
        return (tmp_18, tmp_19)
        