import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = [3, 4, 4, 5, 6, 10, 10, 11, 12, 14]
Y = [16, 21, 25, 29, 22, 19, 21, 25, 21, 25]
data = list(zip(X, Y))

inertias = []
for i in range(1, 11):
    km = KMeans(n_clusters=i)
    km.fit(data)
    inertias.append(km.inertia_)

plt.plot(range(1, 11), inertias, marker='.')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

km = KMeans(n_clusters=2)
km.fit(data)
plt.show()
