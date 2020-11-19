from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from csv import DictReader
import matplotlib.pyplot as plt
import numpy as np

varieties = list()


def fill_cluster(first, second, third):
    global varieties
    varieties.clear()
    with open('users_info.csv', 'r') as read_obj:
        csv_dict_reader = DictReader(read_obj)
        try:
            if third == '':
                for row in csv_dict_reader:
                    if row['Age'] != '' and row['City'] != '' and row['City_id'] != '':
                        varieties.append([int(row[first]), int(row[second])])
            else:
                for row in csv_dict_reader:
                    if row['Age'] != '' and row['City'] != '' and row['City_id'] != '':
                        varieties.append([int(row[first]), int(row[second]), int(row[third])])
        except:
            pass


def get_cluster(cluster_method):
    x = np.array(varieties)
    dendrogram = sch.dendrogram(sch.linkage(x, method=cluster_method))
    plt.show()


def main():
    fill_cluster('Photos', 'Age', '')
    get_cluster('ward')
    fill_cluster('Age', 'City_id', 'Profile_entries')
    get_cluster('average')


if __name__ == "__main__":
    main()
