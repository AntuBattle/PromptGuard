from spacy.lang.en.examples import sentences

from promptguard.nlp_filter import NLPFilter


def test_benign_prompt():
    nlpfilter = NLPFilter()
    sample_sentence = sentences[0]
    assert nlpfilter.safe_eval(sample_sentence) == sample_sentence
