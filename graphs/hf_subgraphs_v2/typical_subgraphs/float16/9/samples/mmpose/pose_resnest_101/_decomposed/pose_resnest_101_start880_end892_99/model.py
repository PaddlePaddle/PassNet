import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, in_0, in_1):
        in_1 += in_0;  in_2 = in_1;  in_1 = in_0 = None
        tmp_18 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        tmp_19 = torch.conv_transpose2d(tmp_18, w_0, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_18 = w_0 = None
        tmp_20 = torch.nn.functional.batch_norm(tmp_19, w_1, w_2, w_4, w_3, False, 0.1, 1e-05);  tmp_19 = w_1 = w_2 = w_4 = w_3 = None
        tmp_21 = torch.nn.functional.relu(tmp_20, inplace = True);  tmp_20 = None
        tmp_22 = torch.conv_transpose2d(tmp_21, w_5, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_21 = w_5 = None
        tmp_23 = torch.nn.functional.batch_norm(tmp_22, w_6, w_7, w_9, w_8, False, 0.1, 1e-05);  tmp_22 = w_6 = w_7 = w_9 = w_8 = None
        tmp_24 = torch.nn.functional.relu(tmp_23, inplace = True);  tmp_23 = None
        tmp_25 = torch.conv_transpose2d(tmp_24, w_10, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_24 = w_10 = None
        tmp_26 = torch.nn.functional.batch_norm(tmp_25, w_11, w_12, w_14, w_13, False, 0.1, 1e-05);  tmp_25 = w_11 = w_12 = w_14 = w_13 = None
        tmp_27 = torch.nn.functional.relu(tmp_26, inplace = True);  tmp_26 = None
        conv2d = torch.conv2d(tmp_27, w_16, w_15, (1, 1), (0, 0), (1, 1), 1);  tmp_27 = w_16 = w_15 = None
        return (conv2d,)
        