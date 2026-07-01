import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def plot_topic_space(lda, dictionary, corpus, tokens):
    # get topic vectors
    doc_vectors = []

    for doc in corpus:
        vec = [prob for _, prob in lda.get_document_topics(doc, minimum_probability=0)]
        doc_vectors.append(vec)

    doc_vectors = np.array(doc_vectors)

    # input vector
    bow = dictionary.doc2bow(tokens)
    input_vec = np.array([
        prob for _, prob in lda.get_document_topics(bow, minimum_probability=0)
    ])

    # PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(doc_vectors)
    input_2d = pca.transform([input_vec])

    # dominant topic = cluster
    labels = np.argmax(doc_vectors, axis=1)

    fig, ax = plt.subplots()

    ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.4
    )

    # input point
    ax.scatter(
        input_2d[:, 0],
        input_2d[:, 1],
        color='red',
        s=30
    )
        # 🔽 get real label from dominant topic
    input_topic = np.argmax(input_vec)
    topic_words = lda.print_topic(input_topic).split('+')
    topic_words = [w.split('*')[1].replace('"', '') for w in topic_words[:2]]
    label = ", ".join(topic_words)

    # 🔽 arrow + real label
    ax.annotate(
        label,
        xy=(input_2d[0, 0], input_2d[0, 1]),
        xytext=(input_2d[0, 0] + 0.4, input_2d[0, 1] + 0.4),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=9,
        color="red"
    )

    # labels
    used_positions = []

    for i in range(lda.num_topics):
        points = reduced[labels == i]
        if len(points) < 10:
            continue

        center = points.mean(axis=0)

        # avoid overlap
        for prev in used_positions:
            if abs(center[0] - prev[0]) < 0.1 and abs(center[1] - prev[1]) < 0.1:
                center[1] += 0.15

        used_positions.append(center)

        topic_words = lda.print_topic(i).split('+')
        topic_words = [w.split('*')[1].replace('"', '') for w in topic_words[:2]]
        label = ", ".join(topic_words)

        ax.text(
            center[0],
            center[1],
            label,
            fontsize=8,  # 🔽 smaller font
            weight='bold'
            # ❌ removed white background box
        )

    ax.set_title("Topic Clusters")
    return fig