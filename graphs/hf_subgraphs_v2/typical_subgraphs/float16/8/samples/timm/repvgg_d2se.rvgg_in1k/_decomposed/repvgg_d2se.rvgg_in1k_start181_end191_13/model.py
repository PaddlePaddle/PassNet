import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor):
        tmp_8 = in_8 + in_9;  in_8 = in_9 = None
        tmp_8 += in_10;  tmp_9 = tmp_8;  tmp_8 = in_10 = None
        tmp_10 = tmp_9.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_10, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_10 = in_1 = in_0 = None
        tmp_12 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_12, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  tmp_12 = in_3 = in_2 = None
        tmp_14 = conv2d_1.sigmoid();  conv2d_1 = None
        tmp_15 = tmp_9 * tmp_14;  tmp_9 = tmp_14 = None
        tmp_16 = torch.nn.functional.relu(tmp_15, inplace = True);  tmp_15 = None
        tmp_17 = torch.nn.functional.batch_norm(tmp_16, in_4, in_5, in_7, in_6, False, 0.1, 1e-05);  in_4 = in_5 = in_7 = in_6 = None
        return (tmp_16, tmp_17)
        