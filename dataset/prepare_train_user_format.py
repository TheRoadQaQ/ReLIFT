from datasets import load_dataset
import pandas as pd

dataset = load_dataset("RoadQAQ/OpenR1", split="train")

print(dataset[0])

ret_dict = []
for item in dataset:
    ret_dict.append(item)

train_df = pd.DataFrame(ret_dict)
train_df.to_parquet("./train_data/openr1.parquet")