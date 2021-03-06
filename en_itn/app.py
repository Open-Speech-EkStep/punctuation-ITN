import streamlit as st
from run_en_punct_itn import post_process_text
from nemo.collections.nlp.models import PunctuationCapitalizationModel

MODEL_PATH = '/path/to/punctuation_en_bert.nemo'


@st.cache(allow_output_mutation=True)
def load_model():
    #model = PunctuationCapitalizationModel.restore_from(MODEL_PATH)
    model = PunctuationCapitalizationModel.from_pretrained('punctuation_en_bert')
    return model


st.title('ITN for English')
text_input = st.text_area('Unpunctuated Text')
model = load_model()
updated_sentence = post_process_text(sent=text_input, model=model)
st.text_area('Punctuated Text', updated_sentence)
