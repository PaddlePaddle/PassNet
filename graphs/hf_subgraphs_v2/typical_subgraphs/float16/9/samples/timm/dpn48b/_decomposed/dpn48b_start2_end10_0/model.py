import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, in_0 : torch.Tensor):
        tmp_9 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        tmp_10 = torch.nn.functional.max_pool2d(tmp_9, 3, 2, 1, 1, ceil_mode = False, return_indices = False);  tmp_9 = None
        tmp_11 = torch.nn.functional.batch_norm(tmp_10, w_4, w_5, w_7, w_6, False, 0.1, 0.001);  w_4 = w_5 = w_7 = w_6 = None
        tmp_12 = torch.nn.functional.relu(tmp_11, inplace = True);  tmp_11 = None
        conv2d = torch.conv2d(tmp_12, w_8, None, (1, 1), (0, 0), (1, 1), 1);  tmp_12 = w_8 = None
        tmp_14 = conv2d[(slice(None, None, None), slice(None, 64, None), slice(None, None, None), slice(None, None, None))]
        tmp_15 = conv2d[(slice(None, None, None), slice(64, None, None), slice(None, None, None), slice(None, None, None))];  conv2d = None
        tmp_16 = torch.nn.functional.batch_norm(tmp_10, w_0, w_1, w_3, w_2, False, 0.1, 0.001);  tmp_10 = w_0 = w_1 = w_3 = w_2 = None
        return (tmp_16, tmp_14, tmp_15)
        