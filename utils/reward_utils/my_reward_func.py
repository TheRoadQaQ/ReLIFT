try:
    from ..openmathinst_utils import extract_answer, math_equal
except:
    from utils.openmathinst_utils import extract_answer, math_equal

import ray
from ray.exceptions import GetTimeoutError
from math_verify import verify, parse
from typing import Union

import signal
from contextlib import contextmanager

# 自定义一个超时异常
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """
    一个上下文管理器，用于限制代码块的执行时间。
    如果在指定时间内没有完成，则会抛出 TimeoutException。
    注意：这个方法在 Windows 上不可用。
    """
    def signal_handler(signum, frame):
        raise TimeoutException(f"Timed out after {seconds} seconds")

    # 设置信号处理函数
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    
    # 设置闹钟
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # 取消闹钟
        signal.alarm(0)
        # 恢复旧的信号处理函数
        signal.signal(signal.SIGALRM, old_handler)


def math_equal_ray(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    timeout: float = 10.0,
    check_antlr_version: bool = True
) -> bool:
    return math_equal(prediction, reference, include_percentage, tolerance, timeout, check_antlr_version)

def verify_ray(
    gold, 
    target, 
    float_rounding: int=6,
    numeric_precision: int=15,
    strict: bool=True,
    timeout_seconds: int=3
) -> bool:
    return verify(gold, target, float_rounding, numeric_precision, strict, timeout_seconds)

def reward_func(data_source, solution_str, ground_truth, extra_info) -> float:
    if "</think>" in solution_str:
        # to avoid the case that boxed appears in both the thinking and the solution
        solution_str = solution_str.split("</think>")[-1]
        format_correct = True
    else:
        # format reward
        format_correct = False

    omi_pred = None
    omi_correct = False
    mathv_pred = None
    mathv_correct = False
    if format_correct:
        # omi
        try:
            omi_pred = extract_answer(solution_str, extract_from_boxed=True)
            omi_correct_ref = math_equal_ray.remote(omi_pred, ground_truth, check_antlr_version=False)
            omi_correct = ray.get(omi_correct_ref, timeout=10.0)
        except GetTimeoutError as e:
            ray.cancel(omi_correct_ref, force=True)
            omi_correct = False
        except Exception:
            omi_correct = False

        # math
        try:
            mathv_pred = parse(solution_str)
            mathv_correct_ref = verify_ray.remote(parse(f"\\boxed{{${ground_truth}$}}"), mathv_pred)
            mathv_correct = ray.get(mathv_correct_ref, timeout=10.0)
        except GetTimeoutError as e:
            ray.cancel(mathv_correct_ref, force=True)
            mathv_correct = False
        except Exception:
            mathv_correct = False

    acc = format_correct and (omi_correct or mathv_correct)
    score = 1.0 if acc else -1.0

    return {
        "score": score,
        "acc": acc,
        "pred": omi_pred,
        "omi_correct": omi_correct,
        "mathv_correct": mathv_correct
    }

def my_reward_func(data_source, solution_str, ground_truth, extra_info) -> float:
    omi_pred = None
    omi_correct = False
    mathv_pred = None
    mathv_correct = False
    
    # omi
    try:
        omi_pred = extract_answer(solution_str, extract_from_boxed=True)
        omi_correct_ref = math_equal(omi_pred, ground_truth, check_antlr_version=False)
        omi_correct = omi_correct_ref
    except Exception:
        omi_correct = False

    # math
    try:
        mathv_pred = parse(solution_str)
        mathv_correct_ref = verify_ray(parse(f"\\boxed{{${ground_truth}$}}"), mathv_pred)
        mathv_correct = mathv_correct_ref
    except Exception:
        mathv_correct = False

    acc = omi_correct or mathv_correct
    score = 1.0 if acc else 0.0

    return {
        "score": score,
        "acc": acc
    }

def my_reward_func_with_timeout(data_source, solution_str, ground_truth, extra_info, timeout_seconds=10) -> float:
    """
    带有超时控制的 Reward 函数。
    
    Args:
        ...
        timeout_seconds (int): 每个验证模块的超时时间（秒）。
    """
    omi_pred = None
    omi_correct = False
    mathv_pred = None
    mathv_correct = False
    
    # --- OMI 验证模块，带超时控制 ---
    try:
        # 将可能卡死的操作放入 time_limit 上下文
        with time_limit(timeout_seconds):
            omi_pred = extract_answer(solution_str, extract_from_boxed=True)
            # 如果 extract_answer 很快，但 math_equal 很慢，也会被捕获
            omi_correct_ref = math_equal(omi_pred, ground_truth, check_antlr_version=False)
            omi_correct = omi_correct_ref
    except TimeoutException:
        # 如果超时，打印一个日志（可选），并将结果视为 False
        print("OMI verification timed out...")
        omi_correct = False
    except Exception:
        # 捕获其他所有异常
        omi_correct = False


    # --- MathV 验证模块，带超时控制 ---
    try:
        # 同样为这个模块设置超时
        with time_limit(timeout_seconds):
            mathv_pred = parse(solution_str)
            mathv_correct_ref = verify_ray(parse(f"\\boxed{{${ground_truth}$}}"), mathv_pred)
            mathv_correct = mathv_correct_ref
    except TimeoutException:
        print("MathV verification timed out...")
        mathv_correct = False
    except Exception:
        mathv_correct = False


    acc = omi_correct or mathv_correct
    score = 1.0 if acc else 0.0

    return {
        "score": score,
        "acc": acc,
        # 你也可以返回更详细的信息用于调试
        "details": {
            "omi_correct": omi_correct,
            "mathv_correct": mathv_correct
        }
    }

if __name__ == "__main__":
    print(my_reward_func("_", "asdada\\boxed{3/2}", "1.50", None))
