import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22):
        in_22 += in_21;  in_23 = in_22;  in_22 = in_21 = None
        tmp_22 = torch.nn.functional.relu(in_23, inplace = True);  in_23 = None
        tmp_23 = torch.conv_transpose2d(tmp_22, in_0, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_22 = in_0 = None
        tmp_24 = torch.nn.functional.batch_norm(tmp_23, in_1, in_2, in_4, in_3, False, 0.1, 1e-05);  tmp_23 = in_1 = in_2 = in_4 = in_3 = None
        tmp_25 = torch.nn.functional.relu(tmp_24, inplace = True);  tmp_24 = None
        tmp_26 = torch.conv_transpose2d(tmp_25, in_5, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_25 = in_5 = None
        tmp_27 = torch.nn.functional.batch_norm(tmp_26, in_6, in_7, in_9, in_8, False, 0.1, 1e-05);  tmp_26 = in_6 = in_7 = in_9 = in_8 = None
        tmp_28 = torch.nn.functional.relu(tmp_27, inplace = True);  tmp_27 = None
        tmp_29 = torch.conv_transpose2d(tmp_28, in_10, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_28 = in_10 = None
        tmp_30 = torch.nn.functional.batch_norm(tmp_29, in_11, in_12, in_14, in_13, False, 0.1, 1e-05);  tmp_29 = in_11 = in_12 = in_14 = in_13 = None
        tmp_31 = torch.nn.functional.relu(tmp_30, inplace = True);  tmp_30 = None
        conv2d = torch.conv2d(tmp_31, in_16, in_15, (1, 1), (0, 0), (1, 1), 1);  tmp_31 = in_16 = in_15 = None
        tmp_33 = torch.flatten(conv2d, 2);  conv2d = None
        linear = torch.nn.functional.linear(tmp_33, in_18, in_17);  in_18 = in_17 = None
        linear_1 = torch.nn.functional.linear(tmp_33, in_20, in_19);  tmp_33 = in_20 = in_19 = None
        return (linear, linear_1)
        