import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"data/fake-kaggle.csv")
print (df)

df.hist()
plt.show()