import pandas as pd

# Load tweets (tab separated)
tweets = pd.read_csv(
    "../data:/rumour_detection/twitter15/source_tweets.txt",
    sep="\t",
    names=["id","text"],
    dtype={"id":str}
)

# Load labels (colon separated)
labels = pd.read_csv(
    "../data:/rumour_detection/twitter15/label.txt",
    sep=":",
    names=["label","id"],
    dtype={"id":str}
)

# Merge
data = tweets.merge(labels,on="id")

print(data.head())
print("Total samples:",len(data))

