from inspect import isclass
import oneflow
from torch import isin
from classes.argdef import ArgDef
from classes.argument import *
from classes.api import *
from classes.database import OneflowDatabase
import json
from utils.printer import dump_data
import random

class OneflowArgument(Argument):
    _supported_types = [
        ArgType.ONEFLOW_DTYPE,
        ArgType.ONEFLOW_OBJECT,
        ArgType.ONEFLOW_TENSOR,
    ]
    _dtypes = [
        oneflow.int8,
        #oneflow.int16,
        oneflow.int32,
        oneflow.int64,
        oneflow.uint8,
        oneflow.float16,
        oneflow.float32,
        oneflow.float64,
        oneflow.bfloat16,
        #oneflow.complex64,
        #oneflow.complex128,
        oneflow.bool,
    ]
    _memory_format = [
        #oneflow.contiguous_format,
        #oneflow.channels_last,
        #oneflow.preserve_format,
    ]
    _float_complex_dtypes = [
        oneflow.float16,
        oneflow.float32,
        oneflow.float64,
        oneflow.bfloat16,
        #oneflow.complex64,
        #oneflow.complex128,
    ]
    _min_values = [0] + [-(1 << i) for i in range(0, 8)]
    _max_values = [(1 << i) - 1 for i in range(0, 8)]
    _tensor_size_limit = 1e7

    def __init__(
        self,
        value,
        type: ArgType,
        shape=None,
        dtype=None,
        max_value=1,
        min_value=0,
    ):
        super().__init__(value, type)
        self.shape = shape
        self.dtype = dtype
        self.max_value = max_value
        self.min_value = min_value
        self.grad = False
        self.new_tensor = True
        
    # workable for tensor    
    def to_code(
        self,
        var_name,
        device="cpu",
        low_precision=False,
        is_sparse=False,
        use_old_tensor=False,
    ) -> str:
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                self.value[i].grad = self.grad
                self.value[i].new_tensor = self.new_tensor
                code += self.value[i].to_code(
                    f"{var_name}_{i}",
                    device=device,
                    low_precision=low_precision,
                    use_old_tensor=use_old_tensor,
                )
                arg_name_list += f"{var_name}_{i},"

            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code
        elif self.type == ArgType.ONEFLOW_TENSOR:
            dtype = self.dtype
            max_value = self.max_value
            min_value = self.min_value
            #if low_precision:
            #    dtype = self.low_precision_dtype(dtype)
            #    max_value, min_value = self.random_tensor_value(dtype)

            # FIXME: shape tune
            shape = self.shape
            size = 1
            for i in range(len(shape) - 1, -1, -1):
                if size * shape[i] > OneflowArgument._tensor_size_limit:
                    shape[i] = 1
                else:
                    size *= shape[i]

            suffix = ""
            if is_sparse:
                suffix += ".to_sparse()"
            if self.grad and self.is_float_or_complex_tensor():
                suffix += ".requires_grad_()"

            code = ""
            if self.new_tensor and not use_old_tensor:
                default_arg = f"{shape}, dtype={dtype}"

                if dtype.is_floating_point:
                    code += f"{var_name}_tensor = oneflow.empty({default_arg})\n"
                    code += f"{var_name}_tensor.uniform_({min_value}, {max_value})\n"
                else:
                    code += f"{var_name}_tensor = oneflow.randint({min_value}, {max_value+1}, {default_arg})\n"
            code += f"{var_name} = {var_name}_tensor.clone().detach().to('{device}'){suffix}\n"
            return code
        elif self.type == ArgType.ONEFLOW_OBJECT:
            return f"{var_name} = {self.value}\n"
        elif self.type == ArgType.ONEFLOW_DTYPE:
            return f"{var_name} = {self.value}\n"
        return super().to_code(var_name)
    
    @staticmethod
    def random_tensor_value(dtype):
        max_value = 1
        min_value = 0
        if dtype == oneflow.bool:
            min_value = choice([0, 1])
            max_value = max(min_value, choice([0, 1]))
        elif dtype == oneflow.uint8:
            min_value = 0
            max_value = choice(OneflowArgument._max_values)
        else:
            min_value = choice(OneflowArgument._min_values)
            max_value = choice(OneflowArgument._max_values)
        return max_value, min_value

    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res is not None:
            return res
        if isinstance(x, oneflow.Tensor):
            return ArgType.ONEFLOW_TENSOR
        elif isinstance(x, oneflow.dtype):
            return ArgType.ONEFLOW_DTYPE
        else:
            return ArgType.ONEFLOW_OBJECT
        
    @staticmethod
    def generate_arg_from_signature(signature):
        """Generate a Oneflow argument from the signature"""
        if signature == "oneflowTensor":
            return OneflowArgument(
                None, ArgType.ONEFLOW_TENSOR, shape=[2, 2], dtype=oneflow.float32
            )
        if signature == "dtype":
            return OneflowArgument(
                choice(OneflowArgument._dtypes), ArgType.ONEFLOW_DTYPE
            )
        if isinstance(signature, str) and signature == "device":
            value = oneflow.device("cpu")
            return OneflowArgument(value, ArgType.ONEFLOW_OBJECT)
        #if isinstance(signature, str) and signature == "torchmemory_format":
        #    value = choice(OneflowArgument._memory_format)
        #    return OneflowArgument(value, ArgType.ONEFLOW_OBJECT)
        if isinstance(signature, str) and signature == "oneflow.strided":
            return OneflowArgument("oneflow.strided", ArgType.ONEFLOW_OBJECT)
        if isinstance(signature, str) and signature.startswith("oneflow."):
            value = eval(signature)
            if isinstance(value, oneflow.dtype):
                return OneflowArgument(value, ArgType.ONEFLOW_DTYPE)
            elif isinstance(value, oneflow.memory_format):
                return OneflowArgument(value, ArgType.ONEFLOW_OBJECT)
            print(signature)
            # TODO:
            assert 0
        if isinstance(signature, bool):
            return OneflowArgument(signature, ArgType.BOOL)
        if isinstance(signature, int):
            return OneflowArgument(signature, ArgType.INT)
        if isinstance(signature, str):
            return OneflowArgument(signature, ArgType.STR)
        if isinstance(signature, float):
            return OneflowArgument(signature, ArgType.FLOAT)
        if isinstance(signature, tuple):
            value = []
            for elem in signature:
                value.append(OneflowArgument.generate_arg_from_signature(elem))
            return OneflowArgument(value, ArgType.TUPLE)
        if isinstance(signature, list):
            value = []
            for elem in signature:
                value.append(OneflowArgument.generate_arg_from_signature(elem))
            return OneflowArgument(value, ArgType.LIST)
        # signature is a dictionary
        
        if isinstance(signature, dict):
            if not (
                "shape" in signature.keys() and "dtype" in signature.keys()
            ):
                raise Exception("Wrong signature {0}".format(signature))
            shape = signature["shape"]
            dtype = signature["dtype"]
            # signature is a ndarray or tensor.
            if isinstance(shape, (list, tuple)):
                if not dtype.startswith("oneflow."):
                    dtype = f"oneflow.{dtype}"
                dtype = eval(dtype)
                max_value, min_value = OneflowArgument.random_tensor_value(dtype)
                return OneflowArgument(
                    None,
                    ArgType.ONEFLOW_TENSOR,
                    shape,
                    dtype=dtype,
                    max_value=max_value,
                    min_value=min_value,
                )
            else:
                return OneflowArgument(
                    None,
                    ArgType.ONEFLOW_TENSOR,
                    shape=[2, 2],
                    dtype=oneflow.float32,
                )
        return OneflowArgument(None, ArgType.NULL)

# self.api = api_name
class OneflowAPI(API):
    def __init__(self, api_name, record=None):
        super().__init__(api_name)
        self.record = (
            OneflowDatabase.get_rand_record(api_name) if record is None else record
        )
        #self.api_name = api_name
        #print(self.record)
        #self.args = OneflowAPI.generate_args_from_record(self.record)
        self.arg_list = []
        self.kwarg_dict = {}
        self.input_arg_list = []
        self.input_kwarg_dict = {}
        params = self.get_info()
        self.handle_params(params) # get input args and kwargs
        self.args = self.generate_args_from_arglist(self.arg_list)
        
        self.is_class = inspect.isclass(eval(self.api))
        print("class",self.is_class)
        #arg_defs = OneflowDatabase.get_argdef(api_name)
        #self.arg_defs = list(map(lambda x: ArgDef.new(x), arg_defs))
        #self.is_class = inspect.isclass(eval(self.api))
    
    def new_record(self, record):
        self.record = record
        #self.args = OneflowAPI.generate_args_from_record(record)
    
    def get_info(self):
        collection = OneflowDatabase.DB[self.api]
        random_input = random.sample(list(collection.find({})), 1)[0]
        #print(random_input)
        return random_input # params
    
    def handle_params(self, params):
        del params['_id']
        arg_list = []
        kwarg_dict = {}
        input_arg_list = []
        input_kwarg_dict = {}
        for item in params.items():
            if item[0].startswith('init_'):
                if item[0].startswith('init_parameter_'):
                    input_arg_list.append(item[1])
                else:
                    item_name = item[0].replace('init_','')
                    input_kwarg_dict[item_name] = item[1]

            else:
                if item[0].startswith('parameter_'):
                    arg_list.append(item[1])
                else:
                    kwarg_dict[item[0]] = item[1]
        print(arg_list, kwarg_dict, input_arg_list, input_kwarg_dict)
        self.arg_list = arg_list
        self.kwarg_dict = kwarg_dict
        self.input_arg_list = input_arg_list
        self.input_kwarg_dict = input_kwarg_dict
    
    def mutate(self, enable_value=True, enable_type=True, enable_db=True):
        if enable_db and do_select_from_db():
            params = self.get_info()
            self.handle_params(params) # get input args and kwargs
            self.args = self.generate_args_from_arglist(self.arg_list)
            do_value_mutation = False
        
    def find_arg_with_name(self, arg_name):
        for arg_def in self.arg_defs:
            if arg_def.name == arg_name:
                return arg_def
        return None    
    '''
    @staticmethod
    def generate_args_from_record(record: dict) -> dict:
        args = {}
        for key, value in record.items():
            if key == "input_signature":
                # NOTE: the input_signature is a dict, which is different than pytorch
                input_args = OneflowAPI.generate_args_from_record(value)
                for k, v in input_args.items():
                    args[f"input_signature_{k}"] = v
            else:
                args[key] = JaxArgument.generate_arg_from_signature(value)
        return args
    '''
    def _get_args(self):
        """
        Return the args, kwargs of API (not including input_signature for class API)
        """
        args = []
        kwargs = {}
        for key, value in self.args.items():
            if key.startswith("parameter:"):
                args.append(value)
            elif key.startswith("input_signature_"):
                pass
            else:
                kwargs[key] = value
        return args, kwargs
    
    def _to_arg_code(self, args, kwargs, prefix="arg", use_old_tensor=False):
        arg_code = ""
        arg_str_list = []
        index = 0
        for arg in args:
            arg_code += arg.to_code(
                f"{prefix}_{index}", use_old_tensor=use_old_tensor
            )
            arg_str_list.append(f"{prefix}_{index}")
            index += 1
        for key, arg in kwargs.items():
            if isinstance(arg, dict) and "content" in arg.keys():
                karg = OneflowArgument.generate_arg_from_signature(key)
            else:
                karg = OneflowArgument.generate_arg_from_signature(arg)
            arg_code += karg.to_code(key, use_old_tensor=use_old_tensor)
            arg_str_list.append(f"{key}={key}")
        return arg_code, ", ".join(arg_str_list)
    
    # fixme
    def to_code(
        self,
        prefix="arg",
        res="res",
        use_try=False,
        error_res="",
        use_old_tensor=False,
    ) -> str:
        #args, kwargs = self._get_args()
        #print(self.args)
        arg_code, arg_str = self._to_arg_code(
            self.args, self.kwarg_dict, prefix=prefix, use_old_tensor=use_old_tensor
        )
        res_code = ""
        if self.is_class and ("Tensor" in self.api):
            res_code = f"{res} = {self.api}({arg_str})\n"
        elif self.is_class:
            # FIXME: I change the instrumentation of input of class
            cls_name = f"{prefix}_class"
            #arg_code += f"{cls_name} = {self.api}({arg_str})\n"
            init_args, init_kwargs = self._get_init_args()
            input_args, input_kwargs = self._get_input_args()
            init_arg_code, init_arg_str = self._to_arg_code(
                init_args, init_kwargs,
                f"{prefix}_input",
                use_old_tensor=use_old_tensor,
            )
            input_arg_code, input_arg_str = self._to_arg_code(
                input_args, input_kwargs,
                f"{prefix}_input",
                use_old_tensor=use_old_tensor,
            )
            arg_code = init_arg_code
            arg_code += f"{cls_name} = {self.api}({init_arg_str})\n"
            arg_code += input_arg_code
            res_code += f"{res} = {cls_name}({input_arg_str})\n"
        else:
            res_code = f"{res} = {self.api}({arg_str})\n"
        invocation = OneflowAPI._to_invocation_code(
            arg_code, res_code, use_try=use_try, error_res=error_res
        )
        return invocation
    
    def _get_input_args(self):
        """
        Return the args, kwargs of input_signature for class API
        """
        args = []
        kwargs = {}
        args = self.generate_args_from_arglist(self.arg_list)
        
        return args, kwargs
    
    def _get_init_args(self):
        """
        Return the args, kwargs of input_signature for class API
        """
        args = []
        kwargs = {}
        for key, value in self.kwarg_dict.items():
            kwargs[key] = value
        return args, kwargs
    
    '''
    @staticmethod
    def generate_args_from_record(record: dict) -> dict:
        args = {}
        for key in record.keys():
            if key != "output_signature":
                args[key] = OneflowArgument.generate_arg_from_signature(
                    record[key]
                )
        return args
    '''
    @staticmethod
    def generate_args_from_arglist(arglist: list) -> list:
        args = []
        for arg in arglist:
            if isinstance(arg, dict) and "content" in arg.keys():
                key = arg["type"]
                args.append(OneflowArgument.generate_arg_from_signature(key))
            else:
                args.append(OneflowArgument.generate_arg_from_signature(arg))
            #args.append(OneflowArgument.generate_arg_from_signature(arg))
        return args
    
    @staticmethod
    def _to_invocation_code(
        arg_code, res_code, use_try=False, error_res="", else_code=""
    ):
        if use_try:
            code = arg_code
            t_code = API.try_except_code(res_code, error_res, else_code)
            code += t_code
        else:
            code = arg_code + res_code
        return code