import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor):
        tmp_8 = in_8.view(32, 608, 48)
        tmp_9 = tmp_8.unsqueeze(1);  tmp_8 = None
        conv2d = torch.conv2d(in_8, in_7, in_6, (1, 1), (0, 0), (1, 1), 1);  in_7 = in_6 = None
        tmp_11 = conv2d.view(32, 1, 48);  conv2d = None
        tmp_12 = torch.nn.functional.softmax(tmp_11, 2, _stacklevel = 5);  tmp_11 = None
        tmp_13 = tmp_12.unsqueeze(-1);  tmp_12 = None
        matmul = torch.matmul(tmp_9, tmp_13);  tmp_9 = tmp_13 = None
        tmp_15 = matmul.view(32, 608, 1, 1);  matmul = None
        conv2d_1 = torch.conv2d(tmp_15, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_15 = in_1 = in_0 = None
        tmp_17 = torch.nn.functional.layer_norm(conv2d_1, (38, 1, 1), in_3, in_2, 1e-05);  conv2d_1 = in_3 = in_2 = None
        tmp_18 = torch.nn.functional.relu(tmp_17, inplace = True);  tmp_17 = None
        conv2d_2 = torch.conv2d(tmp_18, in_5, in_4, (1, 1), (0, 0), (1, 1), 1);  tmp_18 = in_5 = in_4 = None
        tmp_20 = in_8 + conv2d_2;  in_8 = conv2d_2 = None
        return (tmp_20,)
        