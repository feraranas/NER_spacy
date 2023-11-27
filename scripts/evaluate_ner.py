import spacy
from spacy.scorer import Scorer
from spacy.tokens import Doc, DocBin
from spacy.training.example import Example

# make the factory work
from rel_pipe import make_relation_extractor, score_relations

# make the config work
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors

# Load the pre-trained model
ner_model = spacy.load("../training/model-best")

# Load data into memory
doc_bin = DocBin(store_user_data=False).from_disk('../data/predict.spacy')

# examples = [
#     ('Who is Shaka Khan?',
#      {(7, 17, 'PERSON')}),
#     ('I like London and Berlin.',
#      {(7, 13, 'LOC'), (18, 24, 'LOC')})
# ]

def evaluate(ner_model, examples):
    scorer = Scorer()
    example = []
    for input_, annot in examples:
        pred = ner_model(input_)
        print(pred,annot)
        temp = Example.from_dict(pred, dict.fromkeys(annot))
        example.append(temp)
    scores = scorer.score(example)
    return scores

# ner_model = spacy.load('en_core_web_md') # for spaCy's pretrained use 'en_core_web_md'
results = evaluate(ner_model, doc_bin)
print(results)