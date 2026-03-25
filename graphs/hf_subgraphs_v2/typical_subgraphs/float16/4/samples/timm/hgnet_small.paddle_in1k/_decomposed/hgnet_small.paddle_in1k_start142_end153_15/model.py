import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor):
        tmp_5 = torch.nn.functional.relu(in_5, inplace = False);  in_5 = None
        tmp_6 = tmp_5.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_6, in_4, in_3, (1, 1), (0, 0), (1, 1), 1);  tmp_6 = in_4 = in_3 = None
        tmp_8 = torch.sigmoid(conv2d);  conv2d = None
        tmp_9 = torch.mul(tmp_5, tmp_8);  tmp_5 = tmp_8 = None
        tmp_10 = torch.nn.functional.adaptive_avg_pool2d(tmp_9, 1);  tmp_9 = None
        conv2d_1 = torch.conv2d(tmp_10, in_2, None, (1, 1), (0, 0), (1, 1), 1);  tmp_10 = in_2 = None
        tmp_12 = torch.nn.functional.relu(conv2d_1, inplace = False);  conv2d_1 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False);  tmp_12 = None
        tmp_14 = tmp_13.flatten(1, -1);  tmp_13 = None
        linear = torch.nn.functional.linear(tmp_14, in_1, in_0);  tmp_14 = in_1 = in_0 = None
        return (linear,)
        