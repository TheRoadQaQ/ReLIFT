from .run_eval import *

def pm_judge(answer_pred, answer):
    extracted_pred = extract_boxed_content(answer_pred)
    extracted_pred = extracted_pred[0] if len(extracted_pred) > 0 else None
    acc_binary = math_equal(extracted_pred, answer, timeout=True)
    return acc_binary