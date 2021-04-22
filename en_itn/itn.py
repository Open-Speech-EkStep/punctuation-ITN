from nemo_text_processing.inverse_text_normalization.inverse_normalize import inverse_normalize


def perform_itn(sent):
    sent_updated = inverse_normalize(sent, verbose=False)
    return sent_updated


if __name__ == "__main__":
    raw_text = "we paid one hundred and twenty three rupees for this desk, and this."
    print(inverse_normalize(raw_text, verbose=False))
