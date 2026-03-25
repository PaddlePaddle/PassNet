import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_2 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        conv2d = torch.conv2d(tmp_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_2 = in_1 = in_0 = None
        tmp_4 = conv2d.view(12, 1, 2, -1);  conv2d = None
        tmp_5 = tmp_4.transpose(1, 2);  tmp_4 = None
        tmp_6 = torch.nn.functional.softmax(tmp_5, dim = 1);  tmp_5 = None
        tmp_7 = tmp_6.reshape(12, -1);  tmp_6 = None
        tmp_8 = tmp_7.view(12, -1, 1, 1);  tmp_7 = None
        tmp_9 = tmp_8.view(12, 2, -1, 1, 1);  tmp_8 = None
        tmp_10 = tmp_9 * in_3;  tmp_9 = in_3 = None
        tmp_11 = torch.sum(tmp_10, dim = 1);  tmp_10 = None
        tmp_12 = tmp_11.contiguous();  tmp_11 = None
        return (tmp_12,)
        