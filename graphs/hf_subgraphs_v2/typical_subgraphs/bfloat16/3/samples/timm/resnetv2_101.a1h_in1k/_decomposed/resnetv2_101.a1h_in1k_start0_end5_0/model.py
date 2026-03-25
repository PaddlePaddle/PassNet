import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        conv2d = torch.conv2d(in_6, in_5, None, (2, 2), (3, 3), (1, 1), 1);  in_6 = in_5 = None
        tmp_8 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode = False, return_indices = False);  conv2d = None
        tmp_9 = torch.nn.functional.batch_norm(tmp_8, in_1, in_2, in_4, in_3, False, 0.1, 1e-05);  tmp_8 = in_1 = in_2 = in_4 = in_3 = None
        tmp_10 = torch.nn.functional.relu(tmp_9, inplace = True);  tmp_9 = None
        conv2d_1 = torch.conv2d(tmp_10, in_0, None, (1, 1), (0, 0), (1, 1), 1);  in_0 = None
        return (conv2d_1, tmp_10)
        