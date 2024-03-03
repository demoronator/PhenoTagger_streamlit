import streamlit as st
import nltk
from huggingface_hub import hf_hub_download
from tagging_text import bioTag
from dic_ner import dic_ont
from nn_model import bioTag_BERT

@st.cache_resource
def load_model():
    ontfiles={
        'dic_file':'dict/noabb_lemma.dic',
        'word_hpo_file':'dict/word_id_map.json',
        'hpo_word_file':'dict/id_word_map.json'
    }
    biotag_dic = dic_ont(ontfiles)

    vocabfiles={
        'labelfile':'dict/lable.vocab',
        'checkpoint_path':'models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/',
        'lowercase':True
    }
    modelfile='models/pubmedbert_PT_v1.2.h5'
    nn_model=bioTag_BERT(vocabfiles)
    nn_model.load_model(modelfile)

    return biotag_dic, nn_model

sample_text=\
"""Client’s Subjective Concerns/Chief Complaint: “I’m starting to feel more depressed.” Client noted concerns about his mood, endorsing depressed mood, lethargy, insomnia, loss of energy and motivation, and urges to isolate from his romantic partner. 

Clinical Observations: Client appeared disheveled, which is unusual for him, and a marked change since last session. Client sat in a hunched position upon the beginning of the session, and appeared tired, with slowed movements and dysthymic mood. He was attentive and cooperative, and had congruent and appropriate affect. Client denies suicidal ideation. 

Issues and Stressors Discussed/Session Description: Client discussed experiencing increased difficulty with depressive symptoms beginning last week, following an argument with his romantic partner. Client reviewed the details of the argument, and stated that it occurred late on Tuesday night, causing him to get only 4 hours of sleep. Client has continued to attend work, and reports compliance with prescribed medication.

Client stated that the argument is now resolved, and he has been communicating with his partner, though he has had thoughts such as “What’s the point? He doesn’t understand me.” Upon further exploration, this writer learned that, since Tuesday night, the client has been spending more time than usual on his mobile device at night, continuing to get a less than ideal amount of sleep (5-6 hours), and negatively impacting his energy level and mood. While the client did roll his eyes upon this writer bringing up his mobile phone use, he then laughed and stated, “I know, I know, I’m doing it to myself.”"""

footer_text=\
"""---
**PhenoTagger on GitHub**: [Visit GitHub](https://github.com/ncbi-nlp/PhenoTagger)"""

def main():
    st.set_page_config(page_title="PhenoTagger", page_icon="🩺")

    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    
    hf_hub_download(
        repo_id='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        filename='pytorch_model.bin',
        local_dir='models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/'
    )
    hf_hub_download(
        repo_id='lingbionlp/PhenoTagger_v1.2',
        filename='pubmedbert_PT_v1.2.h5',
        local_dir='models/'
    )

    biotag_dic, nn_model = load_model()

    st.title("PhenoTagger")
    text = st.text_area(
        "Enter text here",
        value=sample_text,
        height=650
    )
    if st.button("Run Tagging"):
        result = bioTag(text, biotag_dic, nn_model, onlyLongest=True, abbrRecog=True, Threshold=0.95)
        st.write(result)

    st.markdown(footer_text)

if __name__ == "__main__":
    main()
