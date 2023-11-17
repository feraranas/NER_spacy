from typing import List, Tuple, Callable
import spacy
from spacy.tokens import Doc, Span
from thinc.types import Floats2d, Ints1d, Ragged, cast
from thinc.api import Model, Linear, chain, Logistic
import json
# Import custom components from your rel_model.py file
from scripts.rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors
from scripts.rel_pipe import RelationExtractor, make_relation_extractor

# Register your custom components here
# ... [Insert your custom component registration code]

# Load the trained model
nlp = spacy.load("model-best/")
# nlp.add_pipe("relation_extractor")
# Function to read documents from a .txt file


def read_documents(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        # Splits the file into lines, each line being a document
        return file.read().splitlines()

# Process each document with the SpaCy model


def process_documents(docs: List[str]):
    for doc in docs:
        # Process the document using the SpaCy pipeline
        spacy_doc = nlp(doc)

        # Check and print entities recognized in the document
        print(f"Entities in '{doc}':\n")

        if spacy_doc.ents:
            for ent in spacy_doc.ents:
                print(f"  - {ent.text} ({ent.label_})")
        else:
            print("  No entities found.")

        # Check if any relations are extracted
        if spacy_doc._.rel:
            print("Extracted Relations:")
            for key, relations in spacy_doc._.rel.items():
                # Here key is a tuple of start token indices for the two entities
                start, end = key
                entity1 = spacy_doc[start]
                entity2 = spacy_doc[end]
                print(f"  - Entities: {entity1.text}, {entity2.text}")
                for label, score in relations.items():
                    print(f"    - {label}: {score}")
        else:
            print("  No relations extracted.")

        # Additional debugging or analysis can go here
        print("\n")


# Main execution
if __name__ == "__main__":
    documents = read_documents("test.txt")
    process_documents(documents)
