from nemo.collections.nlp.models import PunctuationCapitalizationModel
from itn import perform_itn
from punctuate_and_capitalize import add_punctuation_and_capitalization


def post_process_text(sent, model):
    if not sent:
        sent = 'Punctuated sentence will appear here'
    itn_sent = perform_itn(sent)
    final_sent = add_punctuation_and_capitalization(itn_sent, model)
    return final_sent


if __name__ == "__main__":
    pass
