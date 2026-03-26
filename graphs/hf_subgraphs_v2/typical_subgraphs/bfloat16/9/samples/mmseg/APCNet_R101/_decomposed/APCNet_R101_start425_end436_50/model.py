import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1, in_2):
        tmp_2 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        tmp_3 = torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'nearest', None);  tmp_2 = None
        tmp_4 = in_1 + tmp_3;  in_1 = tmp_3 = None
        conv2d = torch.conv2d(tmp_4, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_4 = w_1 = w_0 = None
        tmp_6 = conv2d.permute(0, 2, 3, 1);  conv2d = None
        tmp_7 = tmp_6.reshape(1, -1, 9);  tmp_6 = None
        tmp_8 = torch.nn.functional.sigmoid(tmp_7);  tmp_7 = None
        matmul = torch.matmul(tmp_8, in_0);  tmp_8 = in_0 = None
        tmp_10 = matmul.permute(0, 2, 1);  matmul = None
        tmp_11 = tmp_10.contiguous();  tmp_10 = None
        tmp_12 = tmp_11.view(1, 512, 64, 128);  tmp_11 = None
        return (tmp_12,)
        