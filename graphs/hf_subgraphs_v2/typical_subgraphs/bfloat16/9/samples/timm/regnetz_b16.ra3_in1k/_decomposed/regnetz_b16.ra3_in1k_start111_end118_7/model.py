import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor):
        tmp_4 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        tmp_5 = tmp_4.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_5, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_5 = w_1 = w_0 = None
        tmp_7 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_7, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  tmp_7 = w_3 = w_2 = None
        tmp_9 = conv2d_1.sigmoid();  conv2d_1 = None
        tmp_10 = tmp_4 * tmp_9;  tmp_4 = tmp_9 = None
        return (tmp_10,)
        