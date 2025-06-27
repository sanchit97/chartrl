from __future__ import annotations
import os
import torch
import pickle
import json
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Tuple
import numpy as np

from datasets import Dataset

from models import load_vlm_model

dset = "/mnt/home/sanchit/rl-chart/gt-tables/ChartQA/ChartQA Dataset/train/"
model, processor = load_vlm_model("qwen-7b")

# --------------------------------------------------------------------
#  Core helper
# --------------------------------------------------------------------
def auto_rename(df: pd.DataFrame, chart_type: str
               ) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Rename columns of `df` so they match the canonical schema for `chart_type`.
    Returns (df_renamed, mapping_dict).  Raises ValueError if it cannot decide
    unambiguously.

    Supported chart_type keys are the same as in generate_QA().
    """
    df = df.copy(deep=True)          # avoid mutating caller’s frame
    mapping: Dict[str, str] = {}     # original → new name

    # ---------- quick profiles --------------------------------------
    num_cols      = df.select_dtypes(include=[np.number]).columns.tolist()
    dt_cols       = df.select_dtypes(include=['datetime64', 'datetime64[ns]',
                                              'datetimetz']).columns.tolist()
    cat_cols      = [c for c in df.columns if c not in num_cols + dt_cols]
    # sort for stable/ deterministic choices
    num_cols.sort(); dt_cols.sort(); cat_cols.sort()

    # utility lambdas
    def _take(cols: List[str], k: int, role: str) -> List[str]:
        if len(cols) < k:
            raise ValueError(f"Need ≥{k} {role} column(s) for {chart_type}; "
                             f"only {len(cols)} found")
        return cols[:k]

    # ---------- mapping rules --------------------------------------
    ct = chart_type.lower()

    if ct in ("bar", "pie"):
        cat   = _take(cat_cols, 1, "categorical")[0]
        value = _take(num_cols, 1, "numeric")[0]
        mapping = {cat: "category", value: "value"}

    elif ct == "grouped_bar":
        group, category = _take(cat_cols, 2, "categorical")
        value           = _take(num_cols, 1, "numeric")[0]
        mapping = {group: "group", category: "category", value: "value"}

    elif ct == "histogram":
        value = _take(num_cols, 1, "numeric")[0]
        mapping = {value: "value"}

    elif ct == "line":
        x = (_take(dt_cols, 1, "datetime") if dt_cols
             else _take(num_cols, 1, "numeric"))[0]
        y = [c for c in num_cols if c != x][0]  # need second numeric
        mapping = {x: "x", y: "y"}

    elif ct == "scatter":
        x, y = _take(num_cols, 2, "numeric")
        mapping = {x: "x", y: "y"}

    elif ct == "bubble":
        x, y, size = _take(num_cols, 3, "numeric")
        mapping = {x: "x", y: "y", size: "size"}

    elif ct == "line_dual_axis":
        x = (_take(dt_cols, 1, "datetime") if dt_cols
             else _take(num_cols, 1, "numeric"))[0]
        y1, y2 = [c for c in num_cols if c != x][:2]
        mapping = {x: "x", y1: "y1", y2: "y2"}

    elif ct == "scatter_matrix":
        # ≥3 numeric columns – we leave names as-is
        if len(num_cols) < 3:
            raise ValueError("scatter_matrix needs ≥3 numeric columns.")
        mapping = {}                   # no rename

    elif ct == "stacked_bar":
        group, series = _take(cat_cols, 2, "categorical")
        value         = _take(num_cols, 1, "numeric")[0]
        mapping = {group: "group", series: "series", value: "value"}

    elif ct == "multi_line":
        x = (_take(dt_cols, 1, "datetime") if dt_cols
             else _take(num_cols, 1, "numeric"))[0]
        ys = [c for c in num_cols if c != x]
        if len(ys) < 2:
            raise ValueError("multi_line needs ≥2 numeric y-series.")
        mapping = {x: "x"}
        mapping.update({c: f"y_{i}" for i, c in enumerate(ys, 1)})

    else:
        raise ValueError(f"Unsupported chart_type: {chart_type}")

    # ---------- apply mapping --------------------------------------
    if mapping:                         # scatter_matrix may be empty
        df = df.rename(columns=mapping, errors="raise")

    return df, mapping


# ------------------------------------------------------------------ #
#  Helper utilities                                                  #
# ------------------------------------------------------------------ #

def _rank_desc(series: pd.Series) -> List[Any]:
    """Return index labels sorted by descending series value."""
    return series.sort_values(ascending=False).index.tolist()

def _first_cross(x: pd.Series) -> Any:
    """Return x-value where a sign flip first occurs in a series."""
    sign_change = np.sign(x).diff().fillna(0) != 0
    idx = np.where(sign_change)[0]
    return x.index[idx[0]] if idx.size else None

# ------------------------------------------------------------------ #
#  Chart-specific generators                                         #
# ------------------------------------------------------------------ #

def bar_questions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    cat, val = df['category'], df['value']
    q = [
        {"question": "Which bar is tallest?", 
         "answer": cat.iloc[val.idxmax()],
         "not_answer": cat.iloc[val.idxmin()]},
        {"question": "Which bar is shortest?", 
         "answer": cat.iloc[val.idxmin()],
         "not_answer": cat.iloc[val.idxmax()]},
        {"question": "What is the difference between the tallest and shortest bars?",
         "answer": float(val.max() - val.min()),
         "not_answer": float(val.min() - val.max())},
        {"question": "Give the average height of all bars.", 
         "answer": float(val.mean()),
         "not_answer": float(val.mean()+1)},
        {"question": "How many bars exceed the mean height?", 
         "answer": int((val > val.mean()).sum()),
         "not_answer": int((val < val.mean()).sum())},
        {"question": "Rank the bars from highest to lowest.", 
         "answer": _rank_desc(val),
         "not_answer": _rank_desc(val)[::-1]}
    ]
    return q


def pie_questions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    cat, val = df['category'], df['value']
    total = val.sum()
    c0, c1 = cat.unique()[:2]
    pct = (val[cat == c0].iloc[0] / total * 100).round(2)
    combined = (val[cat.isin([c0, c1])].sum() / total * 100).round(2)
    q = [
        {"question": "Which slice is largest?", 
         "answer": cat.iloc[val.idxmax()],
         "not_answer": cat.iloc[val.idxmin()]},
        {"question": f"What percentage of the pie is {c0}?", 
         "answer": float(pct),
         "not_answer": float((100 - pct).round(2))},
        {"question": "What is the second-smallest slice?", 
         "answer": cat.iloc[val.nsmallest(2).idxmax()],
         "not_answer": cat.iloc[val.idxmax()]},
        {"question": "How many slices are below 10 %?", 
         "answer": int((val / total * 100 < 10).sum()),
         "not_answer": int((val / total * 100 > 10).sum())},
        {"question": f"What is the combined share of {c0} and {c1}?",
         "answer": float(combined),
         "not_answer": float((100 - combined).round(2))},
        {"question": "Does any slice exceed 50 % of the pie?",
         "answer": bool((val / total * 100 > 50).any()),
         "not_answer": bool((val / total * 100 <= 50).all())}
    ]
    return q

def grouped_bar_questions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    g, cat, val = df['group'], df['category'], df['value']
    group_means = val.groupby(g).mean()
    group_sums  = val.groupby(g).sum()
    q = []
    # Choose the first group / category deterministically
    g0, c0 = g.unique()[0], cat.unique()[0]
    q.append({"question": f"In group {g0}, which category bar is tallest?",
              "answer": cat[g == g0].iloc[val[g == g0].idxmax()]})
    q.append({"question": f"In group {g0}, what is the total bar height?",
              "answer": float(val[g == g0].sum())})
    q.append({"question": "Across all groups, which category has the highest single bar value?",
              "answer": cat.iloc[val.idxmax()]})
    q.append({"question": "Which group’s average bar height is lowest?",
              "answer": group_means.idxmin()})
    q.append({"question": f"For category {c0}, what is the range across groups?",
              "answer": float((val[cat == c0].groupby(g).max() -
                               val[cat == c0].groupby(g).min()).max())})
    q.append({"question": "List groups ordered by their total bar height.",
              "answer": _rank_desc(group_sums)})
    return q

# -----------------------------------------------------------------
#  Histogram
# -----------------------------------------------------------------
def histogram_questions(df: pd.DataFrame, bins: int = 10) -> List[Dict[str, Any]]:
    v = df['value']
    counts, bin_edges = np.histogram(v, bins=bins)
    max_bin_idx = counts.argmax()
    min_bin_idx = counts.argmin()

    q = [
        {"question": "What is the bin with the most observations?",
         "answer": (float(bin_edges[max_bin_idx]), float(bin_edges[max_bin_idx + 1])),
         "not_answer": (float(bin_edges[min_bin_idx]), float(bin_edges[min_bin_idx + 1]))},
        {"question": "How many values fall below the median?",
         "answer": int((v < v.median()).sum()),
         "not_answer": int((v >= v.median()).sum())},
        {"question": "What is the mean of the data?",
         "answer": float(v.mean()),
         "not_answer": float(v.mean() + 1)},
        {"question": "What is the inter-quartile range?",
         "answer": float(v.quantile(0.75) - v.quantile(0.25)),
         "not_answer": float((v.quantile(0.90) - v.quantile(0.10)))},   # a wider range
        {"question": "How many values are above 1 SD from the mean?",
         "answer": int((v > v.mean() + v.std()).sum()),
         "not_answer": int((v < v.mean() - v.std()).sum())},
        {"question": "Does the distribution look right-skewed?",
         "answer": bool(v.skew() > 0),
         "not_answer": bool(v.skew() <= 0)}
    ]
    return q


# -----------------------------------------------------------------
#  Line
# -----------------------------------------------------------------
def line_questions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    x, y = df['x'], df['y']
    diff  = np.diff(y)
    slope = (y.iloc[-1] - y.iloc[0]) / (x.iloc[-1] - x.iloc[0])
    thr   = y.mean()
    exceed_idx = np.where(y > thr)[0][0] if (y > thr).any() else None
    not_exceed = x.iloc[y.idxmin()] if exceed_idx is not None else x.iloc[0]

    q = [
        {"question": "At which x does the line reach its maximum y?",
         "answer": x.iloc[y.idxmax()],
         "not_answer": x.iloc[y.idxmin()]},
        {"question": "What is the largest single-step increase in y?",
         "answer": float(diff.max()),
         "not_answer": float(diff.min())},
        {"question": f"In which x-interval does y first exceed its mean ({thr:.2f})?",
         "answer": x.iloc[exceed_idx] if exceed_idx is not None else None,
         "not_answer": not_exceed},
        {"question": "What is the average y over the whole series?",
         "answer": float(y.mean()),
         "not_answer": float(y.mean() + 1)},
        {"question": "Is the overall trend positive?",
         "answer": bool(y.iloc[-1] > y.iloc[0]),
         "not_answer": bool(y.iloc[-1] <= y.iloc[0])},
        {"question": "Give the slope between the first and last points.",
         "answer": float(slope),
         "not_answer": float(slope * -1)}
    ]
    return q


# -----------------------------------------------------------------
#  Scatter
# -----------------------------------------------------------------
def scatter_questions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    x, y = df['x'], df['y']
    corr        = df[['x', 'y']].corr().loc['x', 'y']
    origin_dist = ((x ** 2 + y ** 2) ** 0.5)
    frac_above  = ((y > x).sum() / len(df)).round(3)

    q = [
        {"question": "What is the Pearson correlation between x and y?",
         "answer": float(corr),
         "not_answer": float(-corr if corr != 0 else 0.5)},
        {"question": "How many points have x > 0 and y > 0?",
         "answer": int(((x > 0) & (y > 0)).sum()),
         "not_answer": int(((x < 0) & (y < 0)).sum())},
        {"question": "What is the closest point to (0,0)?",
         "answer": int(origin_dist.idxmin()),
         "not_answer": int(origin_dist.idxmax())},
        {"question": "Which point has the highest y?",
         "answer": int(y.idxmax()),
         "not_answer": int(y.idxmin())},
        {"question": "What fraction of points lie above the line y = x?",
         "answer": float(frac_above),
         "not_answer": float(round(1 - frac_above, 3))},
        {"question": "Is the relationship positive or negative?",
         "answer": "positive" if corr > 0 else "negative",
         "not_answer": "negative" if corr > 0 else "positive"}
    ]
    return q


# -----------------------------------------------------------------
#  Bubble
# -----------------------------------------------------------------
def bubble_questions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    x, y, size = df['x'], df['y'], df['size']
    corr_sz_y  = np.corrcoef(size, y)[0, 1]

    q = [
        {"question": "Which bubble is largest?",
         "answer": int(size.idxmax()),
         "not_answer": int(size.idxmin())},
        {"question": "What is the average bubble size?",
         "answer": float(size.mean()),
         "not_answer": float(size.mean() + 1)},
        {"question": "How many bubbles have size below the median?",
         "answer": int((size < size.median()).sum()),
         "not_answer": int((size > size.median()).sum())},
        {"question": "Which bubble has the highest y?",
         "answer": int(y.idxmax()),
         "not_answer": int(y.idxmin())},
        {"question": "Is there a positive relationship between size and y?",
         "answer": bool(corr_sz_y > 0),
         "not_answer": bool(corr_sz_y <= 0)}
    ]
    return q


# -----------------------------------------------------------------
#  Line (dual axis)
# -----------------------------------------------------------------
def line_dual_axis_questions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    x, y1, y2 = df['x'], df['y1'], df['y2']
    gap       = y1 - y2
    cross_x   = _first_cross(gap)
    alt_x     = x.iloc[-1] if cross_x is not None else x.iloc[0]

    q = [
        {"question": "At which x do the two lines cross?",
         "answer": cross_x,
         "not_answer": alt_x},
        {"question": "Which series has a higher final value?",
         "answer": "y1" if y1.iloc[-1] > y2.iloc[-1] else "y2",
         "not_answer": "y2" if y1.iloc[-1] > y2.iloc[-1] else "y1"},
        {"question": "What is the max absolute gap between the two lines?",
         "answer": float(abs(gap).max()),
         "not_answer": float(abs(gap).min())},
        {"question": "Give the mean value of series 1.",
         "answer": float(y1.mean()),
         "not_answer": float(y1.mean() + 1)},
        {"question": "Is series 2 increasing overall?",
         "answer": bool(y2.iloc[-1] > y2.iloc[0]),
         "not_answer": bool(y2.iloc[-1] <= y2.iloc[0])}
    ]
    return q


# -----------------------------------------------------------------
#  Scatter-matrix
# -----------------------------------------------------------------
def scatter_matrix_questions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    num_df    = df.select_dtypes(include=[np.number])
    corr_mtx  = num_df.corr()
    tri       = corr_mtx.where(~np.tril(np.ones(corr_mtx.shape)).astype(bool))
    max_pair  = tri.abs().stack().idxmax()
    min_pair  = tri.abs().stack().idxmin()
    var_series = num_df.var()

    q = [
        {"question": "Which pair of variables has the highest Pearson correlation?",
         "answer": tuple(max_pair),
         "not_answer": tuple(min_pair)},
        {"question": "Which variable shows the largest variance?",
         "answer": var_series.idxmax(),
         "not_answer": var_series.idxmin()},
        {"question": "Is any variable negatively correlated with all others?",
         "answer": bool((corr_mtx.apply(lambda r: (r < 0).all(), axis=1)).any()),
         "not_answer": bool((corr_mtx.apply(lambda r: (r < 0).all(), axis=1)).any()) is False},
        {"question": "How many pairs have |corr| > 0.8?",
         "answer": int((tri.abs() > 0.8).sum().sum()),
         "not_answer": int((tri.abs() <= 0.8).sum().sum())},
        {"question": "List variables ordered by decreasing variance.",
         "answer": _rank_desc(var_series),
         "not_answer": _rank_desc(var_series)[::-1]}
    ]
    return q


# -----------------------------------------------------------------
#  Stacked-bar
# -----------------------------------------------------------------
def stacked_bar_questions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    g, s, val     = df['group'], df['series'], df['value']
    group_total   = val.groupby(g).sum()
    series_total  = val.groupby(s).sum()
    g0, s0        = g.unique()[0], s.unique()[0]
    pct           = (val[(g == g0) & (s == s0)].sum() / group_total[g0] * 100).round(2)

    q = [
        {"question": f"In group {g0}, what is the total stack height?",
         "answer": float(group_total[g0]),
         "not_answer": float(group_total.min())},
        {"question": "Which series contributes most overall across all groups?",
         "answer": series_total.idxmax(),
         "not_answer": series_total.idxmin()},
        {"question": "Which group has the highest total?",
         "answer": group_total.idxmax(),
         "not_answer": group_total.idxmin()},
        {"question": f"For group {g0}, what percentage is series {s0}?",
         "answer": float(pct),
         "not_answer": float(round(100 - pct, 2))},
        {"question": "Which series has the largest single segment?",
         "answer": s.iloc[val.idxmax()],
         "not_answer": s.iloc[val.idxmin()]}
    ]
    return q


# -----------------------------------------------------------------
#  Multi-line
# -----------------------------------------------------------------
def multi_line_questions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    x       = df['x']
    y_cols  = [c for c in df.columns if c.startswith('y')]
    series_max = df[y_cols].max().max()
    line_max   = df[y_cols].max().idxmax()
    line_min   = df[y_cols].max().idxmin()
    final_vals = df[y_cols].iloc[-1]
    slopes     = {c: (df[c].iloc[-1] - df[c].iloc[0]) /
                     (x.iloc[-1] - x.iloc[0]) for c in y_cols}

    q = [
        {"question": "Which line reaches the highest y at any x?",
         "answer": line_max,
         "not_answer": line_min},
        {"question": "At the final x, which line has the lowest value?",
         "answer": final_vals.idxmin(),
         "not_answer": final_vals.idxmax()},
        {"question": "Which line shows the steepest average upward trend?",
         "answer": max(slopes, key=slopes.get),
         "not_answer": min(slopes, key=slopes.get)},
        {"question": "How many lines have a net positive change?",
         "answer": int(sum(v > 0 for v in slopes.values())),
         "not_answer": int(sum(v <= 0 for v in slopes.values()))},
        {"question": "Give the mean value of each line.",
         "answer": {c: float(df[c].mean()) for c in y_cols},
         "not_answer": {c: float(df[c].mean() + 1) for c in y_cols}}
    ]
    return q

# ------------------------------------------------------------------ #
#  Dispatcher to call the right generator                             #
# ------------------------------------------------------------------ #

QUESTION_GENERATORS = {
    "bar": bar_questions,
    "grouped_bar": grouped_bar_questions,
    "pie": pie_questions,
    "histogram": histogram_questions,
    "line": line_questions,
    "scatter": scatter_questions,
    "bubble": bubble_questions,
    "line_dual_axis": line_dual_axis_questions,
    "scatter_matrix": scatter_matrix_questions,
    "stacked_bar": stacked_bar_questions,
    "multi_line": multi_line_questions,
}

def generate_QA(df: pd.DataFrame, chart_type: str) -> List[Dict[str, Any]]:
    """
    Return a list of deterministic (question, answer) dicts 
    for the requested chart type.
    """
    if chart_type not in QUESTION_GENERATORS:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    return QUESTION_GENERATORS[chart_type](df)


def align_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Rename columns to the standard schema used by generate_QA.
    Raises if any keys are missing.
    """
    return df.rename(columns=mapping, errors='raise')


def pick_chart(df):
    numeric_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols  = df.select_dtypes(include=['datetime64', 'datetime64[ns]',
                                               'datetimetz']).columns.tolist()
    categorical_cols = [c for c in df.columns
                        if c not in numeric_cols + datetime_cols]

    n_numeric = len(numeric_cols)
    n_cat     = len(categorical_cols)
    has_time  = len(datetime_cols) > 0

    # Cardinality buckets for categoricals
    LOW_CARD_TH = 12
    cat_low_card  = all(df[c].nunique() <= LOW_CARD_TH for c in categorical_cols)
    cat_high_card = any(df[c].nunique() > LOW_CARD_TH for c in categorical_cols)

    # Constant-sum / part-to-whole flag (pie / 100 % stacked)
    constant_sum = False
    if n_numeric == 1:
        s = df[numeric_cols[0]].sum(skipna=True)
        constant_sum = np.isfinite(s) and np.isclose(s, 100, rtol=0.05)

    # “wide” layout: time or category in the index, each numeric col = series
    wide_layout = (n_numeric > 1 and n_cat == 0)

    if n_numeric == 1:
        # no categoricals → distribution
        if n_cat == 0 and not has_time:
            return "histogram"

        # one categorical axis
        if n_cat == 1 and cat_low_card:
            return "bar"
        if n_cat == 1 and cat_high_card and has_time:
            return "line"

        # two categoricals → grouped bar
        if n_cat == 2:
            return "grouped_bar"

    # -- Two numeric columns --------------------------------------------
    if n_numeric == 2 and not has_time:
        if n_cat == 0:
            return "scatter"
        if n_cat == 1:
            return "bubble"          # scatter with encoded category

    if n_numeric == 2 and has_time:
        return "line_dual_axis"      # two series over time

    # -- ≥3 numeric columns ---------------------------------------------
    if n_numeric >= 3 and n_cat == 0:
        return "scatter_matrix"

    if n_numeric >= 3 and n_cat == 1 and cat_low_card:
        return "stacked_bar"

    # -- Constant-sum cases (part-to-whole) -----------------------------
    if constant_sum and n_cat == 1:
        return "pie" if cat_low_card else "stacked_bar_100"

    # -- Time-series wide layout ----------------------------------------
    if has_time and wide_layout:
        return "multi_line"

    # --------------------------------------------------------------------
    # Fallback
    # --------------------------------------------------------------------
    return "bar"                           # not safe to guess


def get_table(path):
    # Replace 'your_file.csv' with the path to your CSV
    df = pd.read_csv(path)
    return df

def get_plot(path):
    # Replace 'your_file.csv' with the path to your CSV
    img = Image.open(path)
    return img


def pref_format(example):
    # Prepare the input for the chat template
    prompt = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": example["question"]}],
        },
    ]
    chosen = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["label"]}],
        },
    ]
    rejected = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["not_answer"]}],
        },
    ]
    # Apply the chat template
    prompt = processor.apply_chat_template(prompt, tokenize=False)
    chosen = processor.apply_chat_template(chosen, tokenize=False)
    rejected = processor.apply_chat_template(rejected, tokenize=False)
    # Resize the image to ensure it fits within the maximum allowable
    # size of the processor to prevent OOM errors.
    # max_size = self.processor.image_processor.size["longest_edge"]
    # example["image"].thumbnail((max_size, max_size))
    return {"images": [example["image"]], "prompt": prompt, "chosen": chosen, "rejected": rejected}

def main():
    tables = os.listdir(dset+"tables")
    plots = os.listdir(dset+"png")
    # cnt = 0

    all_data = []
    err = 0
    for table,plot in tqdm(zip(tables,plots), total=len(tables)):
        tab = get_table(dset+"tables/"+table)
        chart_type = pick_chart(tab)
        chart = get_plot(dset+"png/"+plot)
        try:
            tab,legend = auto_rename(tab, chart_type)
            questions = generate_QA(tab, chart_type)
            for qs in questions:
                all_data.append({
                    "image": chart,
                    "question": qs["question"],
                    "label": str(qs["answer"]),
                    "not_answer": str(qs["not_answer"]),
                })
        except:
            err+=1
        # print(chart_type)
        # breakpoint()
    print("Charts skipped:", err)
    breakpoint()
    dataset = Dataset.from_list(all_data)
    pref_dataset = dataset.map(pref_format, num_proc=32)
    pref_dataset.save_to_disk("./pref-data/base-qwen-7b-hf-dset-tabular")


main()