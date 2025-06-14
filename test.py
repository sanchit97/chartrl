import os
import json
import pickle

def main():
    data = []
    with open("./pref-data/base-qwen-7b", "rb") as f:
        pref_data = pickle.load(f)

    for batch in pref_data:
        # breakpoint()
        data.append(
            {
                "image": batch[0]["image"],
                "query": batch[0]["query"],
                "label": batch[0]["label"],
                "human": batch[0]['human_or_machine'],
                "correct": batch[0]["query"],
                "wrong": batch[1],
            }
        )
    with open("./pref-data/base-qwen-7b.json", "w") as f:
        json.dump(data, f, indent=2)

main()