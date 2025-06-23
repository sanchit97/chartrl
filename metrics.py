import re
from typing import List, Tuple
import os
import tqdm

import torch



def exact_match(pred, label):
    tot=0
    for i in range(len(pred)):
        if pred[i].lower() == label[i].lower():
            tot+=1
    return tot


def _norm(text: str) -> str:
    """Case-fold, trim, collapse spaces."""
    return re.sub(r'\s+', ' ', text.strip().lower())

def _is_num(t: str) -> bool:
    try:
        float(t)
        return True
    except ValueError:
        return False

def relaxed_accuracy(
    refs: List[str],
    preds: List[str],
    tol: float = 0.05
) -> Tuple[float, List[int]]:
    """
    Return (accuracy, per-item correctness flags).
    """
    assert len(refs) == len(preds), "Mismatched list lengths"
    flags = []
    for ref, pred in zip(refs, preds):
        r, p = _norm(ref), _norm(pred)
        if _is_num(r):                         # numeric case
            try:
                rn, pn = float(r), float(p)
                ok = (abs(pn - rn) / abs(rn)) <= tol if rn else pn == 0
            except ValueError:                 # pred not numeric → wrong
                ok = False
        else:                                  # string case
            ok = (r == p)
        flags.append(int(ok))

    return sum(flags), flags #/ len(flags) if flags else 0.0, flags

# worse version used by unichart (dont use it)
# def relaxed_accuracy(refs, preds):
#     try:
#         gt = float(refs[0])
#         pred = float(preds[0])
#         return [int(abs(gt - pred) / abs(gt) <= 0.05)]
#     except:
#         return [int(str(refs[0]).lower() == str(preds[0]).lower())]