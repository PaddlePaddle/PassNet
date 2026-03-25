import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_8 = in_0 + in_1;  in_0 = in_1 = None
        tmp_9 = tmp_8.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_9, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_9 = w_1 = w_0 = None
        tmp_11 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_11, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  tmp_11 = w_3 = w_2 = None
        tmp_13 = conv2d_1.sigmoid();  conv2d_1 = None
        tmp_14 = tmp_8 * tmp_13;  tmp_8 = tmp_13 = None
        tmp_15 = torch.nn.functional.relu(tmp_14, inplace = True);  tmp_14 = None
        tmp_16 = torch.nn.functional.batch_norm(tmp_15, w_4, w_5, w_7, w_6, False, 0.1, 1e-05);  w_4 = w_5 = w_7 = w_6 = None
        return (tmp_15, tmp_16)
        