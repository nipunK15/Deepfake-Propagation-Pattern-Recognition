import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../results:/final_output.csv")

counts = data["suspicious"].value_counts()
counts.plot(kind="bar")
plt.xlabel("Suspicious")
plt.ylabel("Count")
plt.title("Abnormal Propagation Patterns")
plt.show()


