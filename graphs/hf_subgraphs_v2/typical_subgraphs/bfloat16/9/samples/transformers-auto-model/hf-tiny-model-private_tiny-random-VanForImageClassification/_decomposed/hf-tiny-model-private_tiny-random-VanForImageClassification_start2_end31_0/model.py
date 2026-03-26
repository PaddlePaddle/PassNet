import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, in_0 : torch.Tensor):
        tmp_28 = torch.nn.functional.batch_norm(in_0, w_22, w_23, w_25, w_24, False, 0.1, 1e-05);  w_22 = w_23 = w_25 = w_24 = None
        conv2d = torch.conv2d(tmp_28, w_9, w_8, (1, 1), (0, 0), (1, 1), 1);  w_9 = w_8 = None
        tmp_30 = torch.nn.functional.gelu(conv2d);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_30, w_3, w_2, (1, 1), (2, 2), (1, 1), 16);  w_3 = w_2 = None
        conv2d_2 = torch.conv2d(conv2d_1, w_1, w_0, (1, 1), (9, 9), (3, 3), 16);  conv2d_1 = w_1 = w_0 = None
        conv2d_3 = torch.conv2d(conv2d_2, w_5, w_4, (1, 1), (0, 0), (1, 1), 1);  conv2d_2 = w_5 = w_4 = None
        tmp_34 = tmp_30 * conv2d_3;  tmp_30 = conv2d_3 = None
        conv2d_4 = torch.conv2d(tmp_34, w_7, w_6, (1, 1), (0, 0), (1, 1), 1);  tmp_34 = w_7 = w_6 = None
        tmp_36 = conv2d_4 + tmp_28;  conv2d_4 = tmp_28 = None
        tmp_37 = w_10.unsqueeze(-1);  w_10 = None
        tmp_38 = tmp_37.unsqueeze(-1);  tmp_37 = None
        tmp_39 = tmp_38 * tmp_36;  tmp_38 = tmp_36 = None
        tmp_40 = in_0 + tmp_39;  in_0 = tmp_39 = None
        tmp_41 = torch.nn.functional.batch_norm(tmp_40, w_18, w_19, w_21, w_20, False, 0.1, 1e-05);  w_18 = w_19 = w_21 = w_20 = None
        conv2d_5 = torch.conv2d(tmp_41, w_14, w_13, (1, 1), (0, 0), (1, 1), 1);  tmp_41 = w_14 = w_13 = None
        conv2d_6 = torch.conv2d(conv2d_5, w_12, w_11, (1, 1), (1, 1), (1, 1), 128);  conv2d_5 = w_12 = w_11 = None
        tmp_44 = torch.nn.functional.gelu(conv2d_6);  conv2d_6 = None
        tmp_45 = torch.nn.functional.dropout(tmp_44, 0.0, False, False);  tmp_44 = None
        conv2d_7 = torch.conv2d(tmp_45, w_16, w_15, (1, 1), (0, 0), (1, 1), 1);  tmp_45 = w_16 = w_15 = None
        tmp_47 = torch.nn.functional.dropout(conv2d_7, 0.0, False, False);  conv2d_7 = None
        tmp_48 = w_17.unsqueeze(-1);  w_17 = None
        tmp_49 = tmp_48.unsqueeze(-1);  tmp_48 = None
        tmp_50 = tmp_49 * tmp_47;  tmp_49 = tmp_47 = None
        tmp_51 = tmp_40 + tmp_50;  tmp_40 = tmp_50 = None
        tmp_52 = tmp_51.flatten(2);  tmp_51 = None
        tmp_53 = tmp_52.transpose(1, 2);  tmp_52 = None
        tmp_54 = torch.nn.functional.layer_norm(tmp_53, (16,), w_27, w_26, 1e-06);  tmp_53 = w_27 = w_26 = None
        tmp_55 = tmp_54.view(1, 56, 56, 16);  tmp_54 = None
        tmp_56 = tmp_55.permute(0, 3, 1, 2);  tmp_55 = None
        return (tmp_56,)
        