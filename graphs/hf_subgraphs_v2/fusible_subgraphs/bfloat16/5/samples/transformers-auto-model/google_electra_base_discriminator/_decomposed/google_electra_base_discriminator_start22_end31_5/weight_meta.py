class Program_weight_tensor_meta_extended_attention_mask_2:
	name = "in_0"
	original_name = "extended_attention_mask_2"
	shape = [4, 1, 1, 512]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = float("-inf")
	std = float("inf")


class Program_weight_tensor_meta_query_layer_1:
	name = "in_1"
	original_name = "query_layer_1"
	shape = [4, 12, 512, 64]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = 0.005
	std = 0.112


class Program_weight_tensor_meta_transpose_3:
	name = "in_2"
	original_name = "transpose_3"
	shape = [4, 12, 64, 512]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = 0.001
	std = 0.113


class Program_weight_tensor_meta_value_layer_1:
	name = "in_3"
	original_name = "value_layer_1"
	shape = [4, 12, 512, 64]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = -0.003
	std = 0.110
