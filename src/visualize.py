import matplotlib.pyplot as plt

def plot_topics(topics, all_topics):
    topics = sorted(topics, key=lambda x: x[1], reverse=True)

    labels = []
    values = []

    for topic_id, prob in topics[:5]:
        words = all_topics[topic_id].split('+')
        words = [w.split('*')[1].replace('"', '') for w in words[:2]]
        label = ", ".join(words)

        labels.append(label)
        values.append(prob * 100)

    fig, ax = plt.subplots()

    ax.barh(labels, values)
    ax.invert_yaxis()

    ax.set_xlabel("Probability (%)")
    ax.set_title("Top Topics")

    return fig