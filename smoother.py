import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv('output/infer/main.csv')
# df_b = pd.read_csv('output/infer/full_preds_b_cnn1_10.csv')
print(df)


def majority_vote(arr):
    # Get the counts of each unique element
    counts = np.bincount(arr)
    # Get highest frequency
    votes = counts.argsort()[-1]
    if votes == 2 and counts.sum() != counts[2]:
        votes = counts.argsort()[-2]
    return votes


def smooth_1(df, zeros=None):
    votes = []
    next_vote = 0
    for i in tqdm(range(len(df))):
        maj = majority_vote(df.iloc[i, :])
        if maj in df.iloc[i - 1, :].values and maj != 0:
            # print(i, maj, df.iloc[i-1:i,:].values)
            votes[-1] = maj
        if not next_vote:
            votes.append(maj)
            # if zeros.iloc[i,:].values == 0:
            #     votes.append(0)
            # else:
            #     votes.append(maj)
        else:
            votes.append(next_vote)
            # if zeros.iloc[i,:].values == 0:
            #     votes.append(0)
            # else:
            #     votes.append(next_vote)
            next_vote = 0
        if i < len(df) - 1:
            if maj in df.iloc[i + 1, :].values and maj != 0:
                next_vote = maj
            else:
                next_vote = 0
        if len(votes) > 3:
            if votes[-2] == maj and maj != 0:
                votes[-1] = maj
                # if zeros.iloc[i, :].values == 0:
                #     votes[-1] = 0
                # else:
                #     votes[-1] = maj
        if len(votes) > 4:
            if votes[-3] == maj and maj != 0:
                votes[-2] = maj
                # if zeros.iloc[i, :].values == 0:
                #     votes[-2] = 0
                # else:
                #     votes[-2] = maj
    return pd.DataFrame(votes)

def smooth_2(df):
    row = df.transpose()
    # print(row)
    nonzero_indices = np.where(df != 0)[0]
    subarrays = []
    subarray_start = nonzero_indices[0]
    # Iterate over nonzero indices
    for i in tqdm(range(1, len(nonzero_indices))):
        subarray_end = nonzero_indices[i - 1] + 1
        if nonzero_indices[i] != subarray_end:
            subarray = df[subarray_start:subarray_end].copy()
            # if len(subarray) < 3:
            #     df.iloc[subarray_start-1,0] = subarray.iloc[-1,0]
            #     df.iloc[subarray_end,0] = subarray.iloc[-1,0]
            # else:
            vote = majority_vote(subarray.values.flatten())
            df.iloc[subarray_start:subarray_end] = vote
            subarrays.append(subarray.values.flatten())
            subarray_start = nonzero_indices[i]
        subarray = row[subarray_start:nonzero_indices[-1] + 1]
        subarrays.append(subarray)
    return df

votes = []
voted = smooth_1(df)
voted.to_csv('output/infer/smooth_1_.csv')
print(np.bincount(voted.values.flatten()))
print(voted.value_counts(normalize=True))


voted = smooth_2(voted)
voted.to_csv('output/infer/smooth_2_.csv')



print(np.bincount(voted.values.flatten()))
print(voted.value_counts(normalize=True))








# for i in range(len(df)):
#     votes.append(majority_vote(df.iloc[i,:]))

# voted = pd.DataFrame(votes)
# print(votes)
# print(df_b.values == 0)
# voted.to_csv('output/infer/voted_uncombined.csv', index=False)
# zeros = np.where(df_b.values == 0)[0]
# voted.iloc[zeros,:] = 0
# print(voted)






# voted = pd.DataFrame(votes.values.reshape((16,3000)))
# voted.to_csv('output/infer/combined.csv')