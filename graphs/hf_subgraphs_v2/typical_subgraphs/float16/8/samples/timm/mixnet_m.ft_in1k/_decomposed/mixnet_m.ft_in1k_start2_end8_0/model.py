import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor):
        tmp_10 = torch.nn.functional.relu(in_10, inplace = True);  in_10 = None
        conv2d = torch.conv2d(tmp_10, in_8, None, (1, 1), (1, 1), (1, 1), 24);  in_8 = None
        tmp_12 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  conv2d = in_0 = in_1 = in_3 = in_2 = None
        tmp_13 = torch.nn.functional.relu(tmp_12, inplace = True);  tmp_12 = None
        conv2d_1 = torch.conv2d(tmp_13, in_9, None, (1, 1), (0, 0), (1, 1), 1);  tmp_13 = in_9 = None
        tmp_15 = torch.nn.functional.batch_norm(conv2d_1, in_4, in_5, in_7, in_6, False, 0.1, 1e-05);  conv2d_1 = in_4 = in_5 = in_7 = in_6 = None
        return (tmp_10, tmp_15)
        