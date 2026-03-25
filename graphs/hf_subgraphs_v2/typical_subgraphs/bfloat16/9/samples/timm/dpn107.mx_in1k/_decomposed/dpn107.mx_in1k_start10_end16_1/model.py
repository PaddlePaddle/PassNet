import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, in_0 : torch.Tensor):
        tmp_10 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        conv2d = torch.conv2d(tmp_10, w_0, None, (1, 1), (0, 0), (1, 1), 1);  tmp_10 = w_0 = None
        tmp_12 = torch.nn.functional.batch_norm(conv2d, w_5, w_6, w_8, w_7, False, 0.1, 0.001);  conv2d = w_5 = w_6 = w_8 = w_7 = None
        tmp_13 = torch.nn.functional.relu(tmp_12, inplace = True);  tmp_12 = None
        conv2d_1 = torch.conv2d(tmp_13, w_9, None, (1, 1), (1, 1), (1, 1), 50);  tmp_13 = w_9 = None
        tmp_15 = torch.nn.functional.batch_norm(conv2d_1, w_1, w_2, w_4, w_3, False, 0.1, 0.001);  conv2d_1 = w_1 = w_2 = w_4 = w_3 = None
        return (tmp_15,)
        