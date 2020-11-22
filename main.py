from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from csv import DictReader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fill_cluster(method, args):
    data = pd.read_csv('users_info.csv')
    new_data = data.dropna(subset=['Age', 'Photos', 'City_id'])
    df = pd.DataFrame(new_data, columns=args)
    df['Age'] = df['Age'].astype(int)
    get_cluster(method, df)


def get_cluster(cluster_method, values):
    x = np.array(values)
    dendrogram = sch.dendrogram(sch.linkage(x, method=cluster_method))
    plt.show()


def main():
    fill_cluster('ward', ['Photos', 'Age'])
    fill_cluster('average', ['Age', 'City_id', 'Profile_entries'])


if __name__ == "__main__":
    main()
