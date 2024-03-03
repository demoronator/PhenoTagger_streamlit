import streamlit as st
from tagging_text import bioTag
from dic_ner import dic_ont
from nn_model import bioTag_BERT

@st.cache_resource
def load_model():
    ontfiles={
        'dic_file':'../dict/noabb_lemma.dic',
        'word_hpo_file':'../dict/word_id_map.json',
        'hpo_word_file':'../dict/id_word_map.json'
    }
    biotag_dic = dic_ont(ontfiles)

    vocabfiles={
        'labelfile':'../dict/lable.vocab',
        'checkpoint_path':'../models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/',
        'lowercase':True
    }
    modelfile='../models/pubmedbert_PT_v1.2.h5'
    nn_model=bioTag_BERT(vocabfiles)
    nn_model.load_model(modelfile)

    return biotag_dic, nn_model


def main():
    biotag_dic, nn_model = load_model()
    
    st.title("BioTagger")
    text = st.text_area("Enter text here")
    if st.button("Tag"):
        result = bioTag(text, biotag_dic, nn_model, onlyLongest=True, abbrRecog=True, Threshold=0.95)
        st.write(result)

if __name__ == "__main__":
    main()