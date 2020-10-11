from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, ward, complete, average
from csv import DictReader
import matplotlib.pyplot as plt
import numpy as np

varieties = list()


def fill_cluster(first, second, third):
    global varieties
    varieties.clear()
    with open('users_info.csv', 'r') as read_obj:
        csv_dict_reader = DictReader(read_obj)
        if third == '':
            for row in csv_dict_reader:
                if row['Age'] != '' and row['City'] != '' and row['City_id'] != '':
                    varieties.append([row[first], row[second]])
        else:
            for row in csv_dict_reader:
                if row['Age'] != '' and row['City'] != '' and row['City_id'] != '':
                    varieties.append([row[first], row[second], row[third]])

def k_mean():
    cluster_data = np.array(varieties)
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label='Some label')
    plt.show()
    kmeans = KMeans(n_clusters=7)
    kmeans.fit(cluster_data)
    print('Cluster centers: ')
    print(kmeans.cluster_centers_)
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=kmeans.labels_, cmap='rainbow')
    plt.show()


def get_clusters_Ward():
    cluster_data = np.array(varieties)
    linkage_array = ward(cluster_data)
    dendrogram(linkage_array)
    ax = plt.gca()
    bounds = ax.get_xbound()
    ax.plot(bounds, [10000, 10000], '--', c='k')  # 7.25 to sample
    ax.plot(bounds, [1100, 1100], '--', c='k')  # 4 to sample
    ax.text(bounds[1], 10000, ' 2 Clusters', va='center', fontdict={'size': 5})
    ax.text(bounds[1], 1100, ' 3 Clusters', va='center', fontdict={'size': 5})
    plt.xlabel("Observation index")
    plt.ylabel("Cluster Distance")
    plt.show()


def get_clusters_complete():
    cluster_data = np.array(varieties)
    linkage_array = complete(cluster_data)
    dendrogram(linkage_array)
    ax = plt.gca()
    bounds = ax.get_xbound()
    ax.plot(bounds, [10000, 10000], '--', c='k')  # 7.25 to sample
    ax.plot(bounds, [1100, 1100], '--', c='k')  # 4 to sample
    ax.text(bounds[1], 10000, ' 2 Clusters', va='center', fontdict={'size': 5})
    ax.text(bounds[1], 1100, ' 3 Clusters', va='center', fontdict={'size': 5})
    plt.xlabel("Observation index")
    plt.ylabel("Cluster Distance")
    plt.show()


def get_clusters_average():
    cluster_data = np.array(varieties)
    linkage_array = average(cluster_data)
    dendrogram(linkage_array)
    ax = plt.gca()
    bounds = ax.get_xbound()
    ax.plot(bounds, [10000, 10000], '--', c='k')  # 7.25 to sample
    ax.plot(bounds, [1100, 1100], '--', c='k')  # 4 to sample
    ax.text(bounds[1], 10000, ' 2 Clusters', va='center', fontdict={'size': 5})
    ax.text(bounds[1], 1100, ' 3 Clusters', va='center', fontdict={'size': 5})
    plt.xlabel("Observation index")
    plt.ylabel("Cluster Distance")
    plt.show()


def main():
    fill_cluster('Photos', 'Age', '')
    get_clusters_complete()
    fill_cluster('Age', 'City_id', 'Photos')
    get_clusters_Ward()


if __name__ == "__main__":
    main()
