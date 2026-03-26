import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor):
        conv2d = torch.conv2d(in_10, in_9, in_8, (1, 1), (0, 0), (1, 1), 192);  in_9 = in_8 = None
        tmp_11 = in_11 + conv2d;  in_11 = conv2d = None
        tmp_12 = tmp_11 + in_10;  tmp_11 = in_10 = None
        tmp_13 = torch.nn.functional.batch_norm(tmp_12, in_4, in_5, in_7, in_6, False, 0.1, 1e-05);  tmp_12 = in_4 = in_5 = in_7 = in_6 = None
        tmp_14 = tmp_13.mean((2, 3), keepdim = True)
        conv2d_1 = torch.conv2d(tmp_14, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_14 = in_1 = in_0 = None
        tmp_16 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        conv2d_2 = torch.conv2d(tmp_16, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  tmp_16 = in_3 = in_2 = None
        tmp_18 = conv2d_2.sigmoid();  conv2d_2 = None
        tmp_19 = tmp_13 * tmp_18;  tmp_13 = tmp_18 = None
        return (tmp_19,)
        