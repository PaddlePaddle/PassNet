import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_4 = torch.nn.functional.relu(in_4, inplace = True);  in_4 = None
        conv2d = torch.conv2d(tmp_4, in_0, None, (1, 1), (1, 1), (1, 1), 1536);  tmp_4 = in_0 = None
        conv2d_1 = torch.conv2d(conv2d, in_1, None, (1, 1), (0, 0), (1, 1), 1);  conv2d = in_1 = None
        tmp_7 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        tmp_8 = torch.nn.functional.adaptive_avg_pool2d(tmp_7, 1);  tmp_7 = None
        tmp_9 = tmp_8.flatten(1, -1);  tmp_8 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.0, False, False);  tmp_9 = None
        linear = torch.nn.functional.linear(tmp_10, in_3, in_2);  tmp_10 = in_3 = in_2 = None
        return (linear,)
        