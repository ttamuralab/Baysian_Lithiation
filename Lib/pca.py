import numpy as np
from sklearn.decomposition import PCA
import pandas as pd


def main(ifname, ofname, work_dir, ndim):
    # read
    data = np.loadtxt(f"{work_dir}/candidate/{ifname}")
    print("data_before_pca.shape =", data.shape)
    df = pd.DataFrame(data)

    # standardization
    dfs = df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    print(dfs.isnull().values.sum())
    dfs = dfs.replace([np.inf, -np.inf], np.nan)
    print(dfs.isnull().values.sum())
    dfs = dfs.dropna(how="any", axis=1)
    print(dfs.isnull().values.sum())

    # pca
    pca = PCA()
    pca.fit(dfs)
    feature = pca.transform(dfs)
    print("feature.shape", feature.shape)

    np.savetxt(
        f"{work_dir}/candidate/{ofname}",
        feature[:, :ndim],
    )
