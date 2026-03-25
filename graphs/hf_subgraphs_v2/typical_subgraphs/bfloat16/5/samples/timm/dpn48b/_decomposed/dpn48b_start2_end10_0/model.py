import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor):
        tmp_9 = torch.nn.functional.silu(in_9, inplace = True);  in_9 = None
        tmp_10 = torch.nn.functional.max_pool2d(tmp_9, 3, 2, 1, 1, ceil_mode = False, return_indices = False);  tmp_9 = None
        tmp_11 = torch.nn.functional.batch_norm(tmp_10, in_4, in_5, in_7, in_6, False, 0.1, 0.001);  in_4 = in_5 = in_7 = in_6 = None
        tmp_12 = torch.nn.functional.relu(tmp_11, inplace = True);  tmp_11 = None
        to = tmp_12.to(torch.bfloat16);  tmp_12 = None
        conv2d = torch.conv2d(to, in_8, None, (1, 1), (0, 0), (1, 1), 1);  to = in_8 = None
        tmp_14 = conv2d[(slice(None, None, None), slice(None, 64, None), slice(None, None, None), slice(None, None, None))]
        tmp_15 = conv2d[(slice(None, None, None), slice(64, None, None), slice(None, None, None), slice(None, None, None))];  conv2d = None
        tmp_16 = torch.nn.functional.batch_norm(tmp_10, in_0, in_1, in_3, in_2, False, 0.1, 0.001);  tmp_10 = in_0 = in_1 = in_3 = in_2 = None
        return (tmp_16, tmp_14, tmp_15)
        