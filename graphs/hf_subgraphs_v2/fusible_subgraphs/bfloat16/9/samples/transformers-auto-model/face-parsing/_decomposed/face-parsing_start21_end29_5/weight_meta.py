class Program_weight_tensor_meta_query_layer:
	name = "in_0"
	original_name = "query_layer"
	shape = [1, 1, 16384, 64]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = -0.053
	std = 0.398


class Program_weight_tensor_meta_transpose_4:
	name = "in_1"
	original_name = "transpose_4"
	shape = [1, 1, 64, 256]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = 0.092
	std = 0.145


class Program_weight_tensor_meta_value_layer:
	name = "in_2"
	original_name = "value_layer"
	shape = [1, 1, 256, 64]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = -0.002
	std = 0.028
