import requirements.txt
from sentence_transformers import SentenceTransformer, util
from elasticsearch import Elasticsearch, helpers
import streamlit as st
import pandas as pd
from PIL import Image

#@st.cache(allow_output_mutation=True)

es = Elasticsearch(['https://localhost:9200'],ca_certs=False, verify_certs=False, http_auth=('elastic', 'g7w=Rtt3UGi02ABWUJAW'))
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
#('multi-qa-distilbert-dot-v1')


def run():
    #image = Image.open('IMDB.PNG')
    #st.image(image, caption='')    
    
    st.title('Búsqueda semántica IMDB')
    ranker = st.sidebar.radio('Opciones:', ["Léxica", "Semántica"], index=0)
    st.text('')
    input_text = []
    comment = st.text_input('Ingrese texto de pelicula a buscar !')
    input_text.append(comment)
    
    df_result = pd.DataFrame()
    df_result['Titulo'] = ''
    df_result['Descripcion'] = ''
    df_result['Link'] = ''

    if st.button('SEARCH'):
        with st.spinner('Searching ......'):
            if input_text is not '':
                #result = []
                print(f'INPUT: ', input_text)
                question_embedding = model.encode(input_text[0])
                if ranker == 'Léxica':
                    print('Busqueda lexica....')
                    bm25 = es.search(index="quora", body={"query": {"match": {"field_traduccion": input_text[0] }}})
                    for hit in bm25['hits']['hits'][0:5]:
                        xtitulo = hit['_source']['field_titulo']
                        xlink   = hit['_source']['field_link']
                        xdescripcion = hit['_source']['field_traduccion']
                        df_result = df_result.append({'Titulo': xtitulo, 'Descripcion': xdescripcion, 'Link': xlink} , ignore_index=True)
                        #result.append(hit['_source']['question'])
                else:
                    print('Busqueda semántica....')
                    sem_search = es.search(index="quora", body={
                          "query": {
                            "script_score": {
                              "query": {
                                "match_all": {}
                              },
                              "script": {
                                "source": "cosineSimilarity(params.queryVector, doc['field_traduccion_vector']) + 1.0",
                                "params": {
                                  "queryVector": question_embedding
                                }
                              }
                            }
                          }
                        })
                    for hit in sem_search['hits']['hits'][0:5]:
                        xtitulo = hit['_source']['field_titulo']
                        xlink   = hit['_source']['field_link']
                        xdescripcion = hit['_source']['field_traduccion']
                        df_result = df_result.append({'Titulo': xtitulo, 'Descripcion': xdescripcion, 'Link': xlink} , ignore_index=True)
                        #result.append(hit['_source']['question'])
                                        
                #for i in df_result:
                st.dataframe(df_result)
                    #st.success(f"{str(i)}")

if __name__ == '__main__':
    #model_embedding, client = load_es()
    run()
