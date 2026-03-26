import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor, in_29 : torch.Tensor, in_30 : torch.Tensor, in_31 : torch.Tensor, in_32 : torch.Tensor, in_33 : torch.Tensor, in_34 : torch.Tensor, in_35 : torch.Tensor):
        tmp_36 = in_0[1]
        tmp_37 = in_0[0];  in_0 = None
        tmp_38 = in_6.index_select(-2, tmp_37);  in_6 = tmp_37 = None
        tmp_39 = tmp_36.view((-1, 1));  tmp_36 = None
        tmp_40 = tmp_39.expand_as(tmp_38);  tmp_39 = None
        tmp_41 = tmp_38.new_zeros((1000, 128))
        tmp_42 = tmp_41.scatter_add_(0, tmp_40, tmp_38);  tmp_41 = tmp_40 = tmp_38 = None
        tmp_43 = tmp_42 + in_5;  tmp_42 = in_5 = None
        linear = torch.nn.functional.linear(tmp_43, in_2, in_1);  in_2 = in_1 = None
        tmp_45 = tmp_43 + linear;  tmp_43 = linear = None
        linear_1 = torch.nn.functional.linear(in_35, in_34, in_33);  in_35 = in_34 = in_33 = None
        tmp_47 = torch.nn.functional.dropout(linear_1, p = 0.0, training = False);  linear_1 = None
        tmp_48 = tmp_45 + tmp_47;  tmp_45 = None
        linear_2 = torch.nn.functional.linear(tmp_47, in_4, in_3);  tmp_47 = in_4 = in_3 = None
        tmp_50 = tmp_48 + linear_2;  tmp_48 = linear_2 = None
        tmp_51 = tmp_50.relu_();  tmp_50 = None
        linear_3 = torch.nn.functional.linear(tmp_51, in_8, in_7);  tmp_51 = in_8 = in_7 = None
        tmp_53 = torch.nn.functional.relu(linear_3, inplace = False);  linear_3 = None
        tmp_54 = torch.nn.functional.batch_norm(tmp_53, in_17, in_18, in_20, in_19, False, 0.1, 1e-05);  tmp_53 = in_17 = in_18 = in_20 = in_19 = None
        tmp_55 = torch.nn.functional.dropout(tmp_54, p = 0.0, training = False);  tmp_54 = None
        linear_4 = torch.nn.functional.linear(tmp_55, in_10, in_9);  tmp_55 = in_10 = in_9 = None
        tmp_57 = torch.nn.functional.relu(linear_4, inplace = False);  linear_4 = None
        tmp_58 = torch.nn.functional.batch_norm(tmp_57, in_21, in_22, in_24, in_23, False, 0.1, 1e-05);  tmp_57 = in_21 = in_22 = in_24 = in_23 = None
        tmp_59 = torch.nn.functional.dropout(tmp_58, p = 0.0, training = False);  tmp_58 = None
        linear_5 = torch.nn.functional.linear(tmp_59, in_12, in_11);  tmp_59 = in_12 = in_11 = None
        tmp_61 = torch.nn.functional.relu(linear_5, inplace = False);  linear_5 = None
        tmp_62 = torch.nn.functional.batch_norm(tmp_61, in_25, in_26, in_28, in_27, False, 0.1, 1e-05);  tmp_61 = in_25 = in_26 = in_28 = in_27 = None
        tmp_63 = torch.nn.functional.dropout(tmp_62, p = 0.0, training = False);  tmp_62 = None
        linear_6 = torch.nn.functional.linear(tmp_63, in_14, in_13);  tmp_63 = in_14 = in_13 = None
        tmp_65 = torch.nn.functional.relu(linear_6, inplace = False);  linear_6 = None
        tmp_66 = torch.nn.functional.batch_norm(tmp_65, in_29, in_30, in_32, in_31, False, 0.1, 1e-05);  tmp_65 = in_29 = in_30 = in_32 = in_31 = None
        tmp_67 = torch.nn.functional.dropout(tmp_66, p = 0.0, training = False);  tmp_66 = None
        linear_7 = torch.nn.functional.linear(tmp_67, in_16, in_15);  tmp_67 = in_16 = in_15 = None
        tmp_69 = torch.nn.functional.dropout(linear_7, p = 0.0, training = False);  linear_7 = None
        return (tmp_69,)
        