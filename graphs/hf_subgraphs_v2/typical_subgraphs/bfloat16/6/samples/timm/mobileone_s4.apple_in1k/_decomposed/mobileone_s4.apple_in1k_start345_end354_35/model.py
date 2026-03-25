import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor):
        in_4 += in_5;  in_6 = in_4;  in_4 = in_5 = None
        in_6 += 0;  tmp_4 = in_6;  in_6 = None
        tmp_6 = tmp_4.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_6, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_6 = in_1 = in_0 = None
        tmp_8 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_8, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  tmp_8 = in_3 = in_2 = None
        tmp_10 = conv2d_1.sigmoid();  conv2d_1 = None
        tmp_11 = tmp_4 * tmp_10;  tmp_4 = tmp_10 = None
        tmp_12 = torch.nn.functional.relu(tmp_11, inplace = True);  tmp_11 = None
        return (tmp_12,)
        