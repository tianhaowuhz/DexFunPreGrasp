import pandas as pd
import scipy.stats
from sklearn.cluster import KMeans

embeddings = pd.read_csv("data/pointnet_pretrain_embeddings.csv")
print(embeddings)


embeddings = embeddings.set_index("code")

kmeans = KMeans(n_clusters=10, random_state=0).fit(embeddings)

embeddings["cluster"] = kmeans.labels_

mapping = pd.read_csv("data/category_mapping.csv")
mapping = mapping.set_index("code")

embeddings["category"] = embeddings.index.map(mapping["category"])

# print(embeddings.pivot_table(index="cluster", columns="category", aggfunc="size", fill_value=0))
# print(embeddings)


# cluster 0 / 4
# cluster 0 is better than 4
# new cluster entropy for old cluster ...
# small entropy -> more confident -> better cluster


statistics = pd.read_csv("data/statistics.csv").set_index("code", drop=True)

print(statistics)
print(embeddings)


def compute_entropy(series: pd.Series) -> float:
    """Compute entropy of a categorical series."""
    counts = series.value_counts()
    return scipy.stats.entropy(counts)


print(statistics.query("geo_label == 0"))
print(statistics["geo_label"].unique())

statistics["cluster"] = embeddings["cluster"]

print(compute_entropy(statistics.query("geo_label == 4")["category"]))
print(compute_entropy(statistics.query("geo_label == 0")["category"]))

print("-" * 80)
print(compute_entropy(statistics.query("geo_label == 4")["cluster"]))
print(compute_entropy(statistics.query("geo_label == 0")["cluster"]))


# make tSNE plot for embeddings
import plotly.express as px
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0, init="pca", learning_rate="auto")
tsne_embedding = tsne.fit_transform(embeddings.filter(like="feat_"))

# can use mouse to highlight specific categories
fig = px.scatter(
    tsne_embedding,
    x=0,
    y=1,
    color=embeddings["category"],
    hover_data={"code": embeddings.index, "category": embeddings["category"]},
)
fig.update_layout(
    title="tSNE Embeddings of PointNet Pretrained Model",
    xaxis_title="tSNE Component 1",
    yaxis_title="tSNE Component 2",
)
fig.write_html("pointnet_pretrain_tsne.html")
