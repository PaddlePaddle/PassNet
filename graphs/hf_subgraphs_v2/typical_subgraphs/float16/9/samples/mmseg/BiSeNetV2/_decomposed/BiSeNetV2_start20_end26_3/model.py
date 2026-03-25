import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, in_1 : torch.Tensor):
        tmp_11 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        conv2d = torch.conv2d(tmp_11, w_4, None, (1, 1), (1, 1), (1, 1), 1);  tmp_11 = w_4 = None
        tmp_13 = torch.nn.functional.batch_norm(conv2d, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  conv2d = w_0 = w_1 = w_3 = w_2 = None
        tmp_14 = torch.nn.functional.relu(tmp_13, inplace = True);  tmp_13 = None
        conv2d_1 = torch.conv2d(in_0, w_9, None, (2, 2), (1, 1), (1, 1), 1);  in_0 = w_9 = None
        tmp_16 = torch.nn.functional.batch_norm(conv2d_1, w_5, w_6, w_8, w_7, False, 0.1, 1e-05);  conv2d_1 = w_5 = w_6 = w_8 = w_7 = None
        return (tmp_14, tmp_16)
        