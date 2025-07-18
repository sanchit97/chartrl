import torch
from datasets import load_dataset
import re


def format_reward(completions, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        # pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern = r"<type>.*?</type>\s*<think>.*?</think>\s*<answer>.*?</answer>"
        # TAG_RE = re.compile(r"<think>\s*[\s\S]+?\s*</think>\s*<answer>\s*[\s\S]+?\s*</answer>\s*\Z",flags=re.DOTALL | re.IGNORECASE,)

        # TAG_RE = re.compile(r"<answer>\s*[\s\S]+?\s*</answer>\s*\Z",flags=re.DOTALL | re.IGNORECASE,)
        
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        # matches = [TAG_RE.search(content) for content in completions]

        rewards = [2.0 if match else 0.0 for match in matches]
        print(completions)
        print("Format rewards:", rewards)
        return rewards

def num_token_reward(completions, **kwargs):
    _PATTERNS = [re.compile(r"<type>"),
             re.compile(r"</type>"),
             re.compile(r"<think>"),
             re.compile(r"</think>"),
             re.compile(r"<answer>"),
             re.compile(r"</answer>")]
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
                reward = int(float(abs(gold_parsed - sol)/sol) <= 0.05)
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
        if len(rationale) > 100:
            reward +=  1.0
        steps = rationale.split("step")
        if len(steps) >= 3:
            reward += 2.0
        
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


