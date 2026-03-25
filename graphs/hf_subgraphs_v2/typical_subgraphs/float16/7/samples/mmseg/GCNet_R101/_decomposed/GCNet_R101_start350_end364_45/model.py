import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor):
        tmp_8 = torch.nn.functional.relu(in_8, inplace = True);  in_8 = None
        tmp_9 = tmp_8.view(32, 512, 4096)
        tmp_10 = tmp_9.unsqueeze(1);  tmp_9 = None
        conv2d = torch.conv2d(tmp_8, in_7, in_6, (1, 1), (0, 0), (1, 1), 1);  in_7 = in_6 = None
        tmp_12 = conv2d.view(32, 1, 4096);  conv2d = None
        tmp_13 = torch.nn.functional.softmax(tmp_12, 2, _stacklevel = 5);  tmp_12 = None
        tmp_14 = tmp_13.unsqueeze(-1);  tmp_13 = None
        matmul = torch.matmul(tmp_10, tmp_14);  tmp_10 = tmp_14 = None
        tmp_16 = matmul.view(32, 512, 1, 1);  matmul = None
        conv2d_1 = torch.conv2d(tmp_16, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_16 = in_1 = in_0 = None
        tmp_18 = torch.nn.functional.layer_norm(conv2d_1, (128, 1, 1), in_3, in_2, 1e-05);  conv2d_1 = in_3 = in_2 = None
        tmp_19 = torch.nn.functional.relu(tmp_18, inplace = True);  tmp_18 = None
        conv2d_2 = torch.conv2d(tmp_19, in_5, in_4, (1, 1), (0, 0), (1, 1), 1);  tmp_19 = in_5 = in_4 = None
        tmp_21 = tmp_8 + conv2d_2;  tmp_8 = conv2d_2 = None
        return (tmp_21,)
        