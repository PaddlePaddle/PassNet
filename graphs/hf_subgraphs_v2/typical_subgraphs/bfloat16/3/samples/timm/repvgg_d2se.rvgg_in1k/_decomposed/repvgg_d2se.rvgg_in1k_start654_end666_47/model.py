import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor):
        tmp_6 = in_6 + in_7;  in_6 = in_7 = None
        tmp_7 = tmp_6.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_7, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  tmp_7 = in_3 = in_2 = None
        tmp_9 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_9, in_5, in_4, (1, 1), (0, 0), (1, 1), 1);  tmp_9 = in_5 = in_4 = None
        tmp_11 = conv2d_1.sigmoid();  conv2d_1 = None
        tmp_12 = tmp_6 * tmp_11;  tmp_6 = tmp_11 = None
        tmp_13 = torch.nn.functional.relu(tmp_12, inplace = True);  tmp_12 = None
        tmp_14 = torch.nn.functional.adaptive_avg_pool2d(tmp_13, 1);  tmp_13 = None
        tmp_15 = tmp_14.flatten(1, -1);  tmp_14 = None
        tmp_16 = torch.nn.functional.dropout(tmp_15, 0.0, False, False);  tmp_15 = None
        linear = torch.nn.functional.linear(tmp_16, in_1, in_0);  tmp_16 = in_1 = in_0 = None
        return (linear,)
        