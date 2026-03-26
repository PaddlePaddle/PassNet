import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(in_0, 1)
        conv2d = torch.conv2d(tmp_4, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_4 = w_1 = w_0 = None
        tmp_6 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_6, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  tmp_6 = w_3 = w_2 = None
        tmp_8 = torch.sigmoid(conv2d_1);  conv2d_1 = None
        tmp_9 = in_0 * tmp_8;  in_0 = tmp_8 = None
        tmp_9 += in_1;  tmp_10 = tmp_9;  tmp_9 = in_1 = None
        return (tmp_10,)
        