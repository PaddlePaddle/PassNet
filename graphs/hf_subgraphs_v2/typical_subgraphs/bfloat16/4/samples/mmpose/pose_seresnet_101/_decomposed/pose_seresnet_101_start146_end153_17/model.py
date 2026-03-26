import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor):
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(in_4, 1)
        conv2d = torch.conv2d(tmp_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_4 = in_1 = in_0 = None
        tmp_6 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_6, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  tmp_6 = in_3 = in_2 = None
        tmp_8 = torch.sigmoid(conv2d_1);  conv2d_1 = None
        tmp_9 = in_4 * tmp_8;  in_4 = tmp_8 = None
        tmp_9 += in_5;  tmp_10 = tmp_9;  tmp_9 = in_5 = None
        return (tmp_10,)
        