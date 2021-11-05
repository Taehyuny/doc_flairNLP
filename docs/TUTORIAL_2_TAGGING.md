# Tutorial 2: Tagging your Text

This is part 2 of the tutorial. It assumes that you're familiar with the
[base types](/resources/docs/TUTORIAL_1_BASICS.md) of this library. Here, we show how to use our pre-trained models to
tag your text.

## Tagging with Pre-Trained Sequence Tagging Models

Let's use a pre-trained model for named entity recognition (NER). 
This model was trained over the English CoNLL-03 task and can recognize 4 different entity
types.

```python
from flair.models import SequenceTagger

tagger = SequenceTagger.load('ner')
```
All you need to do is use the `predict()` method of the tagger on a sentence. This will add predicted tags to the tokens
in the sentence. Lets use a sentence with two named entities:
