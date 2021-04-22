from nemo.collections.nlp.models import PunctuationCapitalizationModel


def add_punctuation_and_capitalization(sent, model):
    return model.add_punctuation_capitalization([sent])[0]


if __name__ == "__main__":
    model = PunctuationCapitalizationModel.restore_from(
        "/home/neeraj/ekstep-speech-recognition/punctuation-ITN/en_itn/punct_model/punctuation_en_bert.nemo")

    # try the model on a few examples
    punctuated_list = model.add_punctuation_capitalization(['how are you', 'whats up'])
    print(punctuated_list)
