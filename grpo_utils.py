import torch
from datasets import load_dataset
import re
import json


def format_reward(completions, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        # pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$" # blocks=2
        # pattern = r"^<type>.*?</type>\s*<think>.*?</think>\s*<answer>.*?</answer>" # blocks=3
        # pattern = r"^<type>.*?</type>\s*<table>.*?</table>\s*<think>.*?</think>\s*<answer>.*?</answer>" # blocks=4
        

        # pattern = r"^<think>.*?<type>*?</type>*?<table>*?</table>.*?</think>\n<answer>.*?</answer>$" # blocks=2
        
        # formatted
        pattern = r"^<think>\n<type>.*?</type>\n<table>.*?</table>.*?</think>\n<answer>.*?</answer>$"
        # TAG_RE = re.compile(r"<think>\s*[\s\S]+?\s*</think>\s*<answer>\s*[\s\S]+?\s*</answer>\s*\Z",flags=re.DOTALL | re.IGNORECASE,)

        # TAG_RE = re.compile(r"<answer>\s*[\s\S]+?\s*</answer>\s*\Z",flags=re.DOTALL | re.IGNORECASE,)
        
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        # matches = [TAG_RE.search(content) for content in completions]

        rewards = [3.0 if match else 0.0 for match in matches]
        print(completions)
        print("Format rewards:", rewards)
        return rewards


def compare_tables(pred, gt):
    reward = 0.0
    for col in range(len(pred["columns"])):
        if pred["columns"][col].lower() == gt["columns"][col].lower():
            reward += float(1/len(pred["columns"]))

    for row in range(len(pred["rows"])):
        if pred["rows"][row] == gt["rows"][row]:
            reward += float(1/len(pred["rows"]))
    return reward

def table_style_reward(completions, table, **kwargs):
    rewards = []
    for completion,tab in zip(completions, table):
        reward = 0.0
        # if "```json" in completion:
        #     reward += 0.5            # no fenced JSON

        tab_struct = completion.split("<table>")[-1].strip().split("</table>")[0].strip()

        if "```json" in tab_struct:
                block = tab_struct.split("```json")[1].split("```")[0].strip()
        else:
            block = tab_struct.strip("\n").strip()

        # print(block)
        try:
            # print("SUCCESS PARSING JSON")
            # print("*"*20)
            # print(block)
            jj = json.loads(block.strip("\n").strip())
            # print("Parsed JSON:", jj)
            reward += 1.0            # parseable JSON
            # print("*"*20)
            gt_tab = json.loads(tab.split("```json")[-1].split("```")[0])
            # print("Parsed JSON:", gt_tab)
            reward += 2.0
            # For correctness, compare the tables - maximize reward
            reward += compare_tables(jj, gt_tab) # custom function to compare tables
        except Exception:
            # print("Failed to parse JSON")
            reward += 0.0            # not parseable JSON
        try:
            if set(jj) != {"columns", "rows"}:
                reward += 0.5            # wrong keys
        except:
            reward += 0.0
        rewards.append(reward)
    print("Table style rewards:", rewards)
    return rewards

def num_token_reward(completions, **kwargs):
    _PATTERNS = [
            re.compile(r"<type>"),
            re.compile(r"</type>"),
            re.compile(r"<think>"),
            re.compile(r"</think>"),
            re.compile(r"<answer>"),
            re.compile(r"</answer>"),
            re.compile(r"<table>"),
            re.compile(r"</table>")
            ]
    rewards = []
    for completion in completions:
        reward = 0.0
        reward = int(all(len(p.findall(completion)) == 1 for p in _PATTERNS))
        rewards.append(reward)
    print("Count rewards:", rewards)
    return rewards


def accuracy_reward(completions, label: list[str], **kwargs):
    """Reward function that checks if the completion matches the ground truth.
    - If both gold and prediction are parseable → use math verification.
    - If not parseable → compare as normalized text.
    """
    rewards = []

    # print(completions)

    for completion, sol in zip(completions, label):
        # print(completion)
        # print(label)
        try:
            # print("Completion: ", completion)
            if "<answer>" not in completion:
                gold_parsed = []
            elif "<think>" not in completion:
                gold_parsed = []
            else:
                gold_parsed = completion.split("<answer>")[-1].strip().split("</answer>")[0].strip()
            # print("GOLD PARSED:", gold_parsed)
        except Exception:
            gold_parsed = []

        if len(gold_parsed) != 0:
            # Try parsing predicted answer too
            try:
                sol = float(sol)
                gold_parsed = float(gold_parsed)
                reward = int(float(abs(gold_parsed - sol)/sol) <= 0.01)
                reward = float(reward)
            except Exception as e:
                # print(f"verify failed: {e}, answer: {completion}, gold: {sol}")
                # print(sol, gold_parsed)
                try:
                    reward = int(sol.lower() == gold_parsed.lower())
                    reward = float(reward)
                except:
                    reward = 0.0
        else:
            # fallback to text match
           reward = 0.0

        rewards.append(reward)
    print("Rewards Accuracy:", rewards)
    return rewards


# Ensure that the length of the reasoning is sufficient (50 tokens and 3 steps minimum)
def length_think_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        reward = 0.0
        rationale = completion.split("<think>")[-1].strip().split("</think>")[0].strip()
        if len(rationale) > 200:
            reward +=  1.0
        if len(rationale) > 800:
            reward -=  1.0

        text_rationale = completion.split("</table>")[-1].strip().split("<answer>")[0].strip()
        if len(text_rationale) > 100:
            reward +=  1.0
        if len(text_rationale) > 800:
            reward -=  1.0

        steps = text_rationale.split(".")
        if len(steps) >= 3:
            reward += max(0.1*len(steps), 2.0)
        
        rewards.append(reward)
    print("Length Rewards:", rewards)
    return rewards


def chart_type_reward(completions, chart_type, **kwargs):
    rewards = []
    for completion, c_type in zip(completions, chart_type):
        reward = 0.0
        try:
            pred_type = completion.split("<type>")[-1].strip().split("</type>")[0].strip().lower()
            if pred_type == c_type.lower():
                reward = 1.0
        except Exception as e:
            reward = 0.0
        rewards.append(reward)
    print("Graph Type Rewards:", rewards)
    return rewards


## graph specific rewards here


