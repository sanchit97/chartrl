import torch
from datasets import load_dataset
import re


def format_reward(completions, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        # pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        # pattern = r"*?^<think>.*?</think><answer>.*?</answer>$"



        TAG_RE = re.compile(r"<think>\s*[\s\S]+?\s*</think>\s*<answer>\s*[\s\S]+?\s*</answer>\s*\Z",flags=re.DOTALL | re.IGNORECASE,)

        # TAG_RE = re.compile(r"<answer>\s*[\s\S]+?\s*</answer>\s*\Z",flags=re.DOTALL | re.IGNORECASE,)
        
        # matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        matches = [TAG_RE.search(content) for content in completions]

        rewards = [1.0 if match else 0.0 for match in matches]
        print(completions)
        print("Format rewards:", rewards)
        return rewards

def accuracy_reward(completions, label: list[str], **kwargs):
    """Reward function that checks if the completion matches the ground truth.
    - If both gold and prediction are parseable → use math verification.
    - If not parseable → compare as normalized text.
    """
    rewards = []

    # print(completions)

    for completion, sol in zip(completions, label):
        # breakpoint()
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
                reward = float((abs(gold_parsed - sol))/sol) <= 0.05
                reward = int(reward)
            except Exception as e:
                # print(f"verify failed: {e}, answer: {completion}, gold: {sol}")
                # print(sol, gold_parsed)
                try:
                    reward = float(sol.lower() == gold_parsed.lower())
                    reward = int(reward)
                except:
                    reward = 0.0
        else:
            # fallback to text match
           reward = 0.0

        rewards.append(reward)
    print("Rewards Accuracy:", rewards)
    return rewards