import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_4 = torch.nn.functional.relu(in_4, inplace = True);  in_4 = None
        tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
        conv2d = torch.conv2d(tmp_5, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_5 = in_1 = in_0 = None
        tmp_7 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_7, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  tmp_7 = in_3 = in_2 = None
        tmp_9 = conv2d_1 + 1.0;  conv2d_1 = None
        tmp_10 = tmp_9 / 2.0;  tmp_9 = None
        tmp_11 = tmp_10.clamp_(0.0, 1.0);  tmp_10 = None
        tmp_12 = tmp_4 * tmp_11;  tmp_4 = tmp_11 = None
        return (tmp_12,)
        