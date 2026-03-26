class Program_weight_tensor_meta_query_layer:
	name = "in_0"
	original_name = "query_layer"
	shape = [16, 1, 16384, 64]
	dtype = "torch.float32"
	device = "cuda:0"
	mean = -0.090
	std = 0.334


class Program_weight_tensor_meta_transpose_4:
	name = "in_1"
	original_name = "transpose_4"
	shape = [16, 1, 64, 256]
	dtype = "torch.float32"
	device = "cuda:0"
	mean = 0.085
	std = 0.165


class Program_weight_tensor_meta_value_layer:
	name = "in_2"
	original_name = "value_layer"
	shape = [16, 1, 256, 64]
	dtype = "torch.float32"
	device = "cuda:0"
	mean = -0.005
	std = 0.029
