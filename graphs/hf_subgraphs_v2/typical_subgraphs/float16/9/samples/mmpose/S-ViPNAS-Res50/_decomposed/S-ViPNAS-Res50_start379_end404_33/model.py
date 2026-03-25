import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, in_0, in_1):
        tmp_25 = in_1.view(1, 608, 48)
        tmp_26 = tmp_25.unsqueeze(1);  tmp_25 = None
        conv2d = torch.conv2d(in_1, w_7, w_6, (1, 1), (0, 0), (1, 1), 1);  w_7 = w_6 = None
        tmp_28 = conv2d.view(1, 1, 48);  conv2d = None
        tmp_29 = torch.nn.functional.softmax(tmp_28, 2, _stacklevel = 5);  tmp_28 = None
        tmp_30 = tmp_29.unsqueeze(-1);  tmp_29 = None
        matmul = torch.matmul(tmp_26, tmp_30);  tmp_26 = tmp_30 = None
        tmp_32 = matmul.view(1, 608, 1, 1);  matmul = None
        conv2d_1 = torch.conv2d(tmp_32, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_32 = w_1 = w_0 = None
        tmp_34 = torch.nn.functional.layer_norm(conv2d_1, (38, 1, 1), w_3, w_2, 1e-05);  conv2d_1 = w_3 = w_2 = None
        tmp_35 = torch.nn.functional.relu(tmp_34, inplace = True);  tmp_34 = None
        conv2d_2 = torch.conv2d(tmp_35, w_5, w_4, (1, 1), (0, 0), (1, 1), 1);  tmp_35 = w_5 = w_4 = None
        tmp_37 = in_1 + conv2d_2;  in_1 = conv2d_2 = None
        tmp_37 += in_0;  tmp_38 = tmp_37;  tmp_37 = in_0 = None
        tmp_39 = torch.nn.functional.relu(tmp_38, inplace = True);  tmp_38 = None
        tmp_40 = torch.conv_transpose2d(tmp_39, w_8, None, (2, 2), (1, 1), (0, 0), 16, (1, 1));  tmp_39 = w_8 = None
        tmp_41 = torch.nn.functional.batch_norm(tmp_40, w_9, w_10, w_12, w_11, False, 0.1, 1e-05);  tmp_40 = w_9 = w_10 = w_12 = w_11 = None
        tmp_42 = torch.nn.functional.relu(tmp_41, inplace = True);  tmp_41 = None
        tmp_43 = torch.conv_transpose2d(tmp_42, w_13, None, (2, 2), (1, 1), (0, 0), 16, (1, 1));  tmp_42 = w_13 = None
        tmp_44 = torch.nn.functional.batch_norm(tmp_43, w_14, w_15, w_17, w_16, False, 0.1, 1e-05);  tmp_43 = w_14 = w_15 = w_17 = w_16 = None
        tmp_45 = torch.nn.functional.relu(tmp_44, inplace = True);  tmp_44 = None
        tmp_46 = torch.conv_transpose2d(tmp_45, w_18, None, (2, 2), (1, 1), (0, 0), 16, (1, 1));  tmp_45 = w_18 = None
        tmp_47 = torch.nn.functional.batch_norm(tmp_46, w_19, w_20, w_22, w_21, False, 0.1, 1e-05);  tmp_46 = w_19 = w_20 = w_22 = w_21 = None
        tmp_48 = torch.nn.functional.relu(tmp_47, inplace = True);  tmp_47 = None
        conv2d_3 = torch.conv2d(tmp_48, w_24, w_23, (1, 1), (0, 0), (1, 1), 1);  tmp_48 = w_24 = w_23 = None
        return (conv2d_3,)
        