from classes.oneflow_api import *
from classes.library import Library
from classes.argument import *
from classes.api import *
from os.path import join
import os
from constant.keys import *
import numpy as np

class OneflowLibrary(Library):
    def __init__(self, directory) -> None:
        super().__init__(directory)

    # TODO:
    @staticmethod
    def is_equal(a, b, atol=1e-1, rtol=1e-1, ignore_nan=False, promoted=False):
        def eq_float_tensor(a, b):
            # not strictly equal
            #res = oneflow.equal(x, y)
            #res = res.all().item()
            #print(res)
            a = a.numpy()
            b = b.numpy()
            a[np.isnan(a)] = 0
            b[np.isnan(b)] = 0
            res = (a == b).all()
            return res
        type_a = OneflowArgument.get_type(a)
        type_b = OneflowArgument.get_type(b)
        #print(type_a, type_b)
        if type_a != type_b:
            return False
        if type_a == ArgType.ONEFLOW_TENSOR:
            if a.shape != b.shape:
                return False
            # compare the tensor at cpu
            a = a.clone().cpu()
            b = b.clone().cpu()
            if a.dtype != b.dtype:
                print(promoted)
                if not promoted:
                    return False
                else:
                    promoted_type = oneflow.promote_types(a.dtype, b.dtype)
                    a = a.to(dtype=promoted_type)
                    b = b.to(dtype=promoted_type)
            if not a.dtype.is_floating_point:
                #res = oneflow.eq(a.cpu(), b.cpu()).all()
                #res = res.numpy()
                a = a.numpy()
                b = b.numpy()
                a[np.isnan(a)] = 0
                b[np.isnan(b)] = 0
                res = (a == b).all()
                return res
            #print(a,b)
            return eq_float_tensor(a, b)
        '''
        if type_a == ArgType.JAX_ARRAY:
            a, b = np.asarray(a), np.asarray(b)
            if not a.shape == b.shape:
                return False

            any_nan_a = jnp.any(jnp.isnan(a))
            any_nan_b = jnp.any(jnp.isnan(b))
            if any_nan_a or any_nan_b:
                if any_nan_a != any_nan_b:
                    return False
                elif ignore_nan:
                    return True

            # atol = max(tolerance(a.dtype, atol), tolerance(b.dtype, atol))
            # rtol = max(tolerance(a.dtype, rtol), tolerance(b.dtype, rtol))
            try:
                _assert_numpy_allclose(
                    a, b, atol=atol * a.size, rtol=rtol * b.size
                )
            except Exception as e:
                print(e)
                return False
            else:
                return True
        '''
        if type_a in [ArgType.TUPLE, ArgType.LIST]:
            if len(a) != len(b):
                return False
            equal = True
            for i in range(len(a)):
                equal = equal and OneflowLibrary.is_equal(
                    a[i], b[i], atol, rtol, ignore_nan
                )
            return equal
        elif type_a == ArgType.NULL:
            return True
        else:
            return a == b

    @staticmethod
    def run_code(code):
        results = dict()
        results[ERROR_KEY] = None
        results[ERR_CPU_KEY] = None
        results[ERR_GPU_KEY] = None
        results[ERR_1] = None
        results[ERR_2] = None
        results[GRAD_ERR_1] = None
        results[GRAD_ERR_2] = None
        results[ERR_FN] = None
        results[ERR_CHECK] = None
        error = None
        try:
            exec(code)
        except Exception as e:
            error = str(e)
        return results, error
