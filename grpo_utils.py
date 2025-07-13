import torch
from datasets import load_dataset
import re


def format_reward(completions, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        rewards = [1.0 if match else 0.0 for match in matches]
        return rewards

# def accuracy_reward(completions, solution: list[str], **kwargs):
#     """Reward function that checks if the completion matches the ground truth.
#     - If both gold and prediction are parseable → use math verification.
#     - If not parseable → compare as normalized text.
#     """
#     rewards = []

#     for completion, sol in zip(completions, solution):
#         try:
#             gold_parsed = parse(sol, extraction_mode="first_match")
#         except Exception:
#             gold_parsed = []

#         if len(gold_parsed) != 0:
#             # Try parsing predicted answer too
#             try:
#                 answer_parsed = parse(
#                     completion,
#                     extraction_config=[
#                         LatexExtractionConfig(
#                             normalization_config=NormalizationConfig(
#                                 nits=False,
#                                 malformed_operators=False,
#                                 basic_latex=True,
#                                 boxed="all",
#                                 units=True,
#                             ),
#                             boxed_match_priority=0,
#                             try_extract_without_anchor=False,
#                         )
#                     ],
#                     extraction_mode="first_match",
#                 )
#                 reward = float(verify(gold_parsed, answer_parsed))
#             except Exception as e:
#                 print(f"verify failed: {e}, answer: {completion}, gold: {sol}")
#                 reward = None
#         else:
#             # fallback to text match
#             reward = float(completion.strip().lower() == sol.strip().lower())

#         rewards.append(reward)

#     return rewards