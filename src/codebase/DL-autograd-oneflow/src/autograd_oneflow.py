import os
from classes.api import API
from classes.argument import ArgType, Argument
from classes.database import OneflowDatabase
from classes.argdef import ArgDef
from munkres import Munkres
from classes.oneflow_api import OneflowAPI
from classes.oneflow_library import OneflowLibrary
from constant.keys import *
from utils.loader import load_data
from utils.printer import dump_data
from utils.ad_helper import *
import numpy as np


def is_mismatch_gradcheck(err):
    if err is None:
        return False
    return "Not equal to tolerance" in err

def dump_by_type(res_type, write_code, dir_dict):
    OneflowLibrary.write_to_dir(
        dir_dict[res_type], write_code, output_by_type[res_type]
    )

def domain_check(api):
    check_list = ['oneflow.acos', 'oneflow.acosh', 'oneflow.arccos', 'oneflow.arccosh', 'oneflow.arcsin', 'oneflow.arctanh', 'oneflow.asin', 'oneflow.atanh']
    if api in check_list:
        return True
    else :
        return False

def allow_error(err):
    _allow_errors = [
        # 'grad requires real-valued outputs (output dtype that is a sub-dtype of np.floating), but got int32. For differentiation of functions with integer outputs, use jax.vjp directly',
        # 'grad requires real-valued outputs (output dtype that is a sub-dtype of np.floating), but got complex64.'
        "grad requires real-valued outputs (output dtype that is a sub-dtype of np.floating)",
        "Non-hashable static arguments are not supported, as this can lead to unexpected cache-misses",
        "not implemented",
        "not supported",
    ]
    for a_err in _allow_errors:
        if a_err in err:
            return True
    return False
"""
Config:
    output_success (bool): whether output the program that succeeds
    output_fail (bool): whether output the program that fails
    ...
    allowed_errors: consists of the error message that is allowed
"""
output_success = False
output_fail = False
output_neq = True
output_err = True
output_bug = True

output_by_type = {
    ResultType.SUCCESS: True,
    ResultType.FAIL: True,
    ResultType.BUG: True,
    ResultType.BUG_NORMAL: True,
    ResultType.ERROR: True,
    ResultType.NEQ_STATUS: True,
    ResultType.NEQ_VALUE: True,
    ResultType.NOT_EQ_BK: True,
    ResultType.NOT_EQ_GRAD: True,
    ResultType.NOT_EQUIVALENT: True,
    ResultType.SKIP: False,
}

PossibleValue = {
    ArgType.INT: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    ArgType.FLOAT: [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
    ArgType.STR: Argument._str_values,
    ArgType.BOOL: [True, False],
    ArgType.NULL: [None],
}

OneflowDatabase.database_config("127.0.0.1", 27017, "oneflowDB") #oneflowDB
collist = OneflowDatabase.get_api_list()
#print(collist)

def get_write_code(code):
    #write_code = "from jax.config import config\n"
    #write_code += "config.update('jax_enable_x64', True)\n"
    write_code = "import oneflow\n"
    write_code += "results = dict()\n"
    write_code += code + "\n"
    write_code += "print(results)\n"
    return write_code

def run_and_check(
    code: str,
    dir_dict: dict = {},
    directed: bool = False,
    check_bk_fail: bool = False,
    check_res_value: bool = True,
    ignore_nan: bool = False,
    is_check_grad: bool = False,
    log_file: str = "log-oneflow.py",
):
    write_code = get_write_code(code)
    #print(write_code)
    dump_data(write_code, log_file)

    res_type = ResultType.FAIL
    res_code = ""
    results, error = OneflowLibrary.run_code(code)
    if error:
        res_type = ResultType.ERROR
        res_code += comment_code(str(error))
    else:
        err1 = results[ERR_1]
        err2 = results[ERR_2]
        if err1 is not None and err2 is not None:
            res_type = ResultType.FAIL
        elif err1 is not None or err2 is not None:
            res_code += comment_code(f"ERR 1: {err1}")
            res_code += comment_code(f"ERR 2: {err2}")
            if directed:
                if err1 or allow_error(err2):
                    res_type = ResultType.FAIL
                else:
                    res_type = ResultType.NEQ_STATUS
            else:
                err = err1 if err1 is not None else err2
                if allow_error(err):
                    res_type = ResultType.FAIL
                else:
                    res_type = ResultType.NEQ_STATUS
            if is_check_grad and is_mismatch_gradcheck(err2):
                res_type = ResultType.NEQ_VALUE
        elif check_res_value:
            res1 = results[RES_1]
            res2 = results[RES_2]
            if OneflowLibrary.is_equal(res1, res2, ignore_nan=ignore_nan):
                res_type = ResultType.SUCCESS
            else:
                res_type = ResultType.NEQ_VALUE
        else:
            res_type = ResultType.SUCCESS

    if res_type == ResultType.SUCCESS and check_bk_fail:
        bk_err = results[ERR_CHECK]
        if bk_err and not allow_error(bk_err):
            res_type = ResultType.BK_FAIL
            res_code += comment_code(f"BK FAIL: {bk_err}")

    write_code += res_code
    dump_by_type(res_type, write_code, dir_dict)
    return res_type

# change dir to your own dir
def is_determined(
    api_name,
    num,
    fuzz = False,
    neq_dir='/home/rehanchy/workspace/neq_dir',
    fail_dir='/home/rehanchy/workspace/fail_dir',
    success_dir='/home/rehanchy/workspace/success_dir',
    err_dir='/home/rehanchy/workspace/err_dir',
    bug_dir='/home/rehanchy/workspace/bug_dir',
    strict_mode=False,
    log_file="log-oneflow-rand.py",
):
    os.makedirs(neq_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)
    os.makedirs(bug_dir, exist_ok=True)
    dir_dict = {
        ResultType.NEQ_VALUE: neq_dir,
        ResultType.NEQ_STATUS: neq_dir,
        ResultType.FAIL: fail_dir,
        ResultType.SUCCESS: success_dir,
        ResultType.ERROR: err_dir,
        ResultType.BUG: bug_dir,
    }
    res = []
    api_A = OneflowAPI(api_name)
    records = OneflowDatabase.get_all_records(api_A.api)
    #print(records)
    for k in range(num):
        if fuzz:
            api_A.new_record(records[0])
            api_A.mutate() # not fully implemented
            print(api_A.arg_list)
        #print(records[k])
        else:
            #api_A.get_invocation(records[k])
            api_A.new_record(records[0])
        code = api_A.to_code(
            res=f"results['{RES_1}']",
            use_try=True,
            error_res=f"results['{ERR_1}']",
        )

        code += api_A.to_code(
            res=f"results['{RES_2}']",
            use_try=True,
            error_res=f"results['{ERR_2}']",
            use_old_tensor=True,
        )
        #print(code)
        #print(api_A.args)
        #args, kwargs = api_A._get_args()
        #for key, value in kwargs.items():
        #    print(value.to_code(var_name="x"))
        #print(args)
        #print(kwargs.items())
        print("success")
        contain_nan = False
        r = run_and_check(code, dir_dict, log_file=log_file)
        res.append(r)
        results, error = OneflowLibrary.run_code(code)
        if error:
            continue
        else:
            err1 = results[ERR_1]
            err2 = results[ERR_2]
            if err1 is not None and err2 is not None:
                res_type = ResultType.FAIL
            elif err1 is not None or err2 is not None:
                res_type = ResultType.FAIL
            else:
                res1 = results[RES_1]
                res2 = results[RES_2]
                if type(res1) == "oneflow.Tensor":
                    res1 = res1.numpy()
                    res2 = res2.numpy()
                    nan_res_1 = np.isnan(res1)
                    nan_res_2 = np.isnan(res2)
                    if type(nan_res_1) == np.ndarray:
                        contain_nan_1 = (True in nan_res_1)
                        contain_nan_2 = (True in nan_res_2)
                    else:
                        contain_nan_1 = (nan_res_1 == True)
                        contain_nan_2 = (nan_res_2 == True)
                    if contain_nan_1 or contain_nan_2:
                        contain_nan = True
                    else:
                        contain_nan = False
                else:
                    contain_nan = False
        
    return res, contain_nan
count = 0
determined = 0
undetermined = 0
failed = 0
nan_list = []
determined_list = []
'''
for i in collist:
    print(i)
    count += 1
print(count)
'''
#fw = open("/home/rehanchy/workspace/success_dir/determined_list.txt", "w")
for i in collist:
    if(count > 3):
        break
    print("start")
    print(count)
    
    print(i)
    
    count += 1
    
    #params = OneflowAPI.get_info(i)
    #OneflowAPI.handle_params(params)
    if(i == "oneflow.as_tensor"):
        determined+=1
        determined_list.append(i)
        continue
    if(i == "oneflow.constant_initializer"):
        determined+=1
        determined_list.append(i)
        continue
    if(i == "oneflow.any"): # instrumentation failure
        continue
    if(i == "oneflow.batch_gather"): # core dumped
        continue
    if(i == "oneflow.bernoulli"): # the problem is about domain
        continue
    if(i == "oneflow.chunk"):   # instrumentation failure (divide 0)
        continue
    if(i == "oneflow.nn.BCELoss"):  # domain issues caused core dumped
        continue    
    if(i == "oneflow.nn.COCOReader"): # instrumentation failure -> no such file or directory
        continue
    if(i == "oneflow.nn.CTCLoss"): # domain issues caused core dumped # Check failed: max_target_length >= target_lengths_ptr[b] (10 vs. 55)
        continue
    if(i == "oneflow.nn.CrossEntropyLoss"): # domain issues caused core dumped
        continue
    if(i == "oneflow.nn.NLLLoss"): # domain issues caused core dumped
        continue
    if(i == "oneflow.nn.functional.cross_entropy"):
        continue
    if(i == "oneflow.nn.functional.ctc_greedy_decoder"): # Check failed: max_input_length >= input_lengths_ptr[b] (20 vs. 22)
        continue
    if(i == "oneflow.gather"): # domain issues caused core dumped
        continue
    if(i == "oneflow.nn.functional.embedding"): # domain issues caused core dumped
        continue
    if(i == "oneflow.nn.functional.one_hot"): # domain issues caused core dumped
        continue
    if(i == "oneflow.nn.functional.sparse_softmax_cross_entropy"):
        continue
    if(i == "oneflow.scatter"): # domain issues caused core dumped  UNIMPLEMENTEDThe index element 3 is out of bounds for dimension 1 with size 2.
        continue
    if(i == "oneflow.save"): # wrong signature
        continue
    if(i == "oneflow.scatter_add"):
        continue
    if(i == "oneflow.size"): # no such API
        continue
    if(i == "oneflow.slice"): # inf loop    
        continue
    if(i == "oneflow.tensor_scatter_nd_update"): 
        continue
    if(i == "oneflow.sort"):
        continue
    if(i == "oneflow.tensor_to_tensor_buffer"):
        continue
    if(i == "oneflow.distributed_partial_fc_sample"): # not implemented error
        continue
    if(i == "oneflow.empty"): #instrumentation failure
        continue
    if(i == "oneflow.from_numpy"):
        continue
    if(i == "oneflow.index_select"):
        continue
    if domain_check(i):
        continue
    res, contain_nan = is_determined(api_name=i , num=1, fuzz=False)
    if contain_nan:
        nan_list.append(i)
    else:
        if res[0] == ResultType.NEQ_VALUE or res[0] == ResultType.NEQ_STATUS:
            undetermined += 1
        if res[0] == ResultType.SUCCESS:
            determined += 1
            determined_list.append(i)
        if res[0] == ResultType.FAIL:
            failed += 1
        print(res)    
        print(f"undetermined: {undetermined}")
        print(f"determined: {determined}")
        print(f"failed: {failed}")
    print(nan_list)
#for j in determined_list:
#    fw.write(j + "\n")
#fw.close()
#print(undetermined)