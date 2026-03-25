import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor):
        tmp_11 = torch.nn.functional.relu(in_11, inplace = True);  in_11 = None
        conv2d = torch.conv2d(in_0, in_5, None, (2, 2), (3, 3), (1, 1), 1);  in_0 = in_5 = None
        tmp_13 = torch.nn.functional.batch_norm(conv2d, in_1, in_2, in_4, in_3, False, 0.1, 1e-05);  conv2d = in_1 = in_2 = in_4 = in_3 = None
        tmp_14 = torch.nn.functional.relu(tmp_13, inplace = True);  tmp_13 = None
        conv2d_1 = torch.conv2d(tmp_14, in_10, None, (2, 2), (1, 1), (1, 1), 1);  tmp_14 = in_10 = None
        tmp_16 = torch.nn.functional.batch_norm(conv2d_1, in_6, in_7, in_9, in_8, False, 0.1, 1e-05);  conv2d_1 = in_6 = in_7 = in_9 = in_8 = None
        return (tmp_11, tmp_16)
        