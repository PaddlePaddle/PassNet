import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor):
        tmp_4 = in_4.mean((2, 3), keepdim = True)
        to = tmp_4.to(torch.bfloat16);  tmp_4 = None
        conv2d = torch.conv2d(to, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  to = in_1 = in_0 = None
        tmp_6 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        to_1 = tmp_6.to(torch.bfloat16);  tmp_6 = None
        conv2d_1 = torch.conv2d(to_1, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  to_1 = in_3 = in_2 = None
        tmp_8 = conv2d_1.sigmoid();  conv2d_1 = None
        tmp_9 = in_4 * tmp_8;  in_4 = tmp_8 = None
        tmp_9 += in_5;  tmp_10 = tmp_9;  tmp_9 = in_5 = None
        return (tmp_10,)
        