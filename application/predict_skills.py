import typer
import spacy
import srsly
import pandas as pd
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
import random
import tqdm

# make the factory work
from rel_pipe import make_relation_extractor, score_relations

# make the config work
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors

# Load the pre-trained model
nlp = spacy.load("../training/model-best")

# Load the jsonl data to predict
data = srsly.read_jsonl("./golden.jsonl")
data_instances = [eg["text"] for eg in data]

# Represent data via DocBin [list of docs]
db = DocBin()
for text in data_instances:
     doc = nlp.make_doc(text)
     db.add(doc)


db.to_disk("predict.spacy")

# Load data into memory
doc_bin = DocBin(store_user_data=True).from_disk('./predict.spacy')
docs = doc_bin.get_docs(nlp.vocab)

def _score_and_format(examples, thresholds):
    for threshold in thresholds:
        r = score_relations(examples, threshold)
        results = {k: "{:.2f}".format(v * 100) for k, v in r.items()}
        print(f"threshold {'{:.2f}'.format(threshold)} \t {results}")

# Make predictions for entities
examples = []
random_examples = []
for gold in docs:
    pred = Doc(
        nlp.vocab,
        words=[t.text for t in gold],
        spaces=[t.whitespace_ for t in gold],
    )
    pred.ents = gold.ents

    for name, proc in nlp.pipeline:
        pred = proc(pred)
    examples.append(Example(pred, gold))

    relation_extractor = nlp.get_pipe("relation_extractor")
    get_instances = relation_extractor.model.attrs["get_instances"]
    for (e1, e2) in get_instances(pred):
        offset = (e1.start, e2.start)
        if offset not in pred._.rel:
            pred._.rel[offset] = {}
        for label in relation_extractor.labels:
            pred._.rel[offset][label] = random.uniform(0, 1)
    random_examples.append(Example(pred, gold))

# print(examples)
print('/////////////////////')
print('/////////////////////')
print('/////////////////////')
print('/////////////////////')
print('/////////////////////')
# print(random_examples)

thresholds = [0.000, 0.050, 0.100, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
print()
print("Random baseline:")
_score_and_format(random_examples, thresholds)
print()
print("Results of the trained model:")
_score_and_format(examples, thresholds)
