import pandas as pd
import matplotlib.pyplot as plt

from _logging.main import get_logger

_logger = get_logger(__name__)

########################################################################################################################
######################################### STATS ANALYSIS ###############################################################
########################################################################################################################

data = pd.read_csv("../data/ecg_data.csv")
# print(data.head())

'''
last column is the label, where abnormal examples are labeled as 1 and normal examples are labeled as 0.
the other columns are the ECG data points (electrical potential) over time
ECG values are of type float and the labels are of type int
'''

'''
# print(data.info())
# 5 rows, 141 columns
# 4997 entries, 0 to 4996
# no null values
'''

# data.describe() on normal ECG
last_column = data.iloc[:, -1]
normal_data = data[last_column.eq(0)]
total_mean = normal_data.iloc[:, :-1].mean().mean()
_logger.info("Total mean across all patients with normal ECG: %s", total_mean)

# data.describe() on abnormal ECG
last_column = data.iloc[:, -1]
normal_data = data[last_column.eq(1)]
total_mean = normal_data.iloc[:, :-1].mean().mean()
_logger.info("Total mean across all patients with abnormal ECG: %s", total_mean)


########################################################################################################################
################################################## HISTOGRAMS ##########################################################
########################################################################################################################

data_no_labels = data.iloc[:, :-1]
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))

# Normal ECG
for i, column in enumerate(data_no_labels.columns[:5]):
    axes[0, i].hist(data_no_labels[column], bins=15)
    axes[0, i].set_title(f"Normal histogram {i + 1}")
    axes[0, i].set_xlabel("ECG data point")
    axes[0, i].set_ylabel("Frequency")

# # Abnormal ECG
for i, column in enumerate(data_no_labels.columns[-5:]):
    axes[1, i].hist(data_no_labels[column], bins=15)
    axes[1, i].set_title(f"Abnormal histogram {i + 1}")
    axes[1, i].set_xlabel("ECG data point")
    axes[1, i].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

'''
Normal ECG data points are centered around 0, with a few outliers compared to the abnormal ECG data points which are more spread out.
'''

