import json

import typer
from pathlib import Path

from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
from wasabi import Printer

msg = Printer()
# occupation -> knows -> skill
# occupation -> has -> skill

SYMM_LABELS = ["KNOWS", "HAS"]

MAP_LABELS = {
    "KNOWS": "knows",
    "HAS": "has",
    "SKILL": "skill",
    "OCC": "occupation",
}

# Define the proportions for train, test, and dev sets
train_ratio = 0.7
test_ratio = 0.15
dev_ratio = 0.15


# MAP_LABELS = {
#     "Pos-Reg": "Regulates",
#     "Neg-Reg": "Regulates",
#     "Reg": "Regulates",
#     "No-rel": "Regulates",
#     "Binds": "Binds",
# }


def main(json_loc: Path, train_file: Path, dev_file: Path, test_file: Path):
    """Creating the corpus from the Prodigy annotations."""
    Doc.set_extension("rel", default={})
    vocab = Vocab()

    docs = {"train": [], "dev": [], "test": []}
    # ids = {"train": set(), "dev": set(), "test": set()}
    count_all = {"train": 0, "dev": 0, "test": 0}
    count_pos = {"train": 0, "dev": 0, "test": 0}

    total_lines = sum(1 for line in open(json_loc, 'r', encoding='utf8'))

    with json_loc.open("r", encoding="utf8") as jsonfile:

        # Initialize counts for each set
        train_count = int(total_lines * train_ratio)
        test_count = int(total_lines * test_ratio)
        dev_count = total_lines - train_count - test_count

        for line in jsonfile:
            example = json.loads(line)
            span_starts = set()
            if example["answer"] == "accept":
                neg = 0
                pos = 0
                try:
                    # Parse the tokens
                    words = [t["text"] for t in example["tokens"]]
                    spaces = [t["ws"] for t in example["tokens"]]
                    doc = Doc(vocab, words=words, spaces=spaces)

                    # Parse the OCC, SKILL entities
                    spans = example["spans"]
                    entities = []
                    span_end_to_start = {}
                    for span in spans:
                        # Agregué esta linea para evitar overlapping de "two labels on the same token"
                        if any(e.label_ == span["label"] for e in entities):
                            continue

                        entity = doc.char_span(
                            span["start"], span["end"], label=span["label"]
                        )
                        span_end_to_start[span["token_end"]] = span["token_start"]
                        entities.append(entity)
                        span_starts.add(span["start"])
                    doc.ents = entities

                    # Parse the relations
                    rels = {}
                    for x1 in span_starts:
                        for x2 in span_starts:
                            rels[(x1, x2)] = {}
                    relations = example["relations"]
                    for relation in relations:
                        # the 'head' and 'child' annotations refer to the end token in the span
                        # but we want the first token
                        start = span_end_to_start[relation["head"]]
                        end = span_end_to_start[relation["child"]]
                        label = relation["label"]
                        if (label == None or not label):
                            label = "KNOWS"
                        label = MAP_LABELS[label]
                        if label not in rels[(start, end)]:
                            rels[(start, end)][label] = 1.0
                            pos += 1
                        if label in SYMM_LABELS:
                            if label not in rels[(end, start)]:
                                rels[(end, start)][label] = 1.0
                                pos += 1

                    # The annotation is complete, so fill in zero's where the data is missing
                    for x1 in span_starts:
                        for x2 in span_starts:
                            for label in MAP_LABELS.values():
                                if label not in rels[(x1, x2)]:
                                    neg += 1
                                    rels[(x1, x2)][label] = 0.0
                    doc._.rel = rels

                    # only keeping documents with at least 1 positive case
                    # could we use this even 0 positive cases?
                    if pos > 0:
                        # use the original PMID/PMCID to decide on train/dev/test split
                        # article_id = example["meta"]["source"]
                        # article_id = article_id.replace("BioNLP 2011 Genia Shared Task, ", "")
                        # article_id = article_id.replace(".txt", "")
                        # article_id = article_id.split("-")[1]
                        # if article_id.endswith("4"):
                        #     ids["dev"].add(article_id)
                        #     docs["dev"].append(doc)
                        #     count_pos["dev"] += pos
                        #     count_all["dev"] += pos + neg
                        # elif article_id.endswith("3"):
                        #     ids["test"].add(article_id)
                        #     docs["test"].append(doc)
                        #     count_pos["test"] += pos
                        #     count_all["test"] += pos + neg
                        # else:
                        #     ids["train"].add(article_id)
                        #     docs["train"].append(doc)
                        #     count_pos["train"] += pos
                        #     count_all["train"] += pos + neg
                        if train_count > 0:
                            docs["train"].append(doc)
                            count_pos["train"] += pos
                            count_all["train"] += pos + neg
                            train_count -= 1
                        elif test_count > 0:
                            docs["test"].append(doc)
                            count_pos["test"] += pos
                            count_all["test"] += pos + neg
                            test_count -= 1
                        elif dev_count > 0:
                            docs["dev"].append(doc)
                            count_pos["dev"] += pos
                            count_all["dev"] += pos + neg
                            dev_count -= 1

                except KeyError as e:
                    problematic_key = e.args[0]
                    problematic_value = example.get(problematic_key, None)
                    msg.fail(f"Skipping doc because of key error: {problematic_key} with value {problematic_value} in {example}")


    docbin = DocBin(docs=docs["train"], store_user_data=True)
    docbin.to_disk(train_file)
    msg.info(
        # f"{len(docs['train'])} training sentences from {len(ids['train'])} articles, "
        f"{len(docs['train'])} training from {train_count} sentences\n"
        f"{count_pos['train']}/{count_all['train']} pos instances."
    )

    docbin = DocBin(docs=docs["dev"], store_user_data=True)
    docbin.to_disk(dev_file)
    msg.info(
        # f"{len(docs['dev'])} dev sentences from {len(ids['dev'])} articles, "
        f"{len(docs['dev'])} dev {dev_count} sentences\n"
        f"{count_pos['dev']}/{count_all['dev']} pos instances."
    )

    docbin = DocBin(docs=docs["test"], store_user_data=True)
    docbin.to_disk(test_file)
    msg.info(
        f"{len(docs['test'])} test {test_count} sentences\n"
        f"{count_pos['test']}/{count_all['test']} pos instances."
    )


if __name__ == "__main__":
    typer.run(main)
