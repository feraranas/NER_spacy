import spacy
from spacy.scorer import Scorer
from spacy.tokens import Doc, DocBin
from spacy.training.example import Example
import srsly

# make the factory work
from rel_pipe import make_relation_extractor, score_relations

# make the config work
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors

# Load the pre-trained model
nlp = spacy.load("../training/model-best")


def evaluate(ner_model, json_file):
    scorer = Scorer()
    example = []

    # Load the data to predict
    data = srsly.read_jsonl(json_file)
    data_instances = [eg["text"] for eg in data]

    # Represent data via DocBin [list of docs]
    db = DocBin()
    for text in data_instances:
        db.add(nlp(text))

    for doc in db.get_docs(ner_model.vocab):
        # pred = doc
    # for input_, annot in examplbes:
        # pred = ner_model(input_)
        # print(pred, pred.ents)
        temp = Example.from_dict(doc, dict.fromkeys(doc.ents))
        example.append(temp)
    scores = scorer.score(example)
    return scores

# ner_model = spacy.load('en_core_web_md') # for spaCy's pretrained use 'en_core_web_md'
results = evaluate(nlp, "../application/golden.jsonl")
print(results)