import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1):
        conv2d = torch.conv2d(in_1, w_9, w_8, (1, 1), (0, 0), (1, 1), 224);  w_9 = w_8 = None
        tmp_11 = in_0 + conv2d;  in_0 = conv2d = None
        tmp_12 = tmp_11 + in_1;  tmp_11 = in_1 = None
        tmp_13 = torch.nn.functional.batch_norm(tmp_12, w_4, w_5, w_7, w_6, False, 0.1, 1e-05);  tmp_12 = w_4 = w_5 = w_7 = w_6 = None
        tmp_14 = tmp_13.mean((2, 3), keepdim = True)
        conv2d_1 = torch.conv2d(tmp_14, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_14 = w_1 = w_0 = None
        tmp_16 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        conv2d_2 = torch.conv2d(tmp_16, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  tmp_16 = w_3 = w_2 = None
        tmp_18 = conv2d_2.sigmoid();  conv2d_2 = None
        tmp_19 = tmp_13 * tmp_18;  tmp_13 = tmp_18 = None
        return (tmp_19,)
        