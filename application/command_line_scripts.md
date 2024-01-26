## Command to run a rel & ner annotation session with:

- a new blank dataset
- with no previous loaded model ner or rel
- a non-anotated jsonl file

```bash
prodigy rel.manual shapingskills-taxonomy blank:en ./assets/singapore_skills_taxonomy_NOT_ANOTATED.jsonl --label KNOWS,HAS --add-ents --span-label OCC,SKILL --wrap
```