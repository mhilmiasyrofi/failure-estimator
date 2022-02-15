import json
import numpy as np
from failure_estimator import HuggingFaceTransformer


def read_data(json_fpath):

    labels = []
    texts = []

    file = open(json_fpath)

    for line in file.readlines():
        js_instance = json.loads(line)

        label = js_instance["label"]
        text = js_instance["text"].lower()

        labels.append(label)
        texts.append(text)

    file.close()

    return texts, labels

def randomize(texts, labels):
    assert len(texts) == len(labels)
    indexs = np.array(list(range(len(texts))))
    np.random.shuffle(indexs)
    indexs = list(indexs)
    ## get array items given a list of indexes
    texts = list(map(texts.__getitem__, indexs))
    labels = list(map(labels.__getitem__, indexs))
    return texts, labels


if __name__ == "__main__" :

    json_fpath = "/media/mhilmiasyrofi/failure-estimator/dummy_corpus.json"
    texts, labels = read_data(json_fpath)

    text, labels = randomize(texts, labels)

    # use the same unique name from huggingface
    estimator = HuggingFaceTransformer(name="bert-base-uncased")

    estimator.fit(X=texts, y=labels)
    

    # try predicting using the same data
    n = 10

    sub_texts = texts[:n]
    sub_labels = labels[:n]

    probability = estimator.predict(sub_texts)

    for l, p in zip(sub_labels, probability):
        print(f"label: {l}, prob: {p:.2f}")

