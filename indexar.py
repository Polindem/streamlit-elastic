from sentence_transformers import SentenceTransformer, util
from elasticsearch import Elasticsearch, helpers
import csv
import os
import tqdm.autonotebook

#es = Elasticsearch(['https://localhost:9200'],http_auth=('elastic', 'g7w=Rtt3UGi02ABWUJAW'))
es = Elasticsearch(['https://localhost:9200'],ca_certs=False, verify_certs=False, basic_auth=('elastic', 'g7w=Rtt3UGi02ABWUJAW'))
model = SentenceTransformer('multi-qa-distilbert-dot-v1')

#def run():
#Datos del archivo Csv a leer
name_file = 'movies_prueba1'	
ext_file = '.csv'
name_file_input = name_file + ext_file
max_corpus_size = 100000

#Prepara data para la indexación
nro_fila = 1
long_cadena = 0
cadena_traduccion = ''
cadena_titulo= ''
cadena_link  = ''
campo_traduccion = {}
campo_titulo = {}
campo_link = {}
num_leidos = 0
encontro_error = 0
with open(name_file_input) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:        
        try:
            cadena_titulo = row['titulo']
            cadena_link = row['link']            
            cadena_traduccion = row['traduccion']
            if (cadena_titulo != '') and (cadena_link != '') and (cadena_traduccion != ''):
                #Extrae doble comilla al inicio y final
                long_cadena = len(cadena_traduccion) - 1
                cadena_traduccion = cadena_traduccion[1:long_cadena]
                #Convierte nro fila a string
                nro_fila_char = str(nro_fila)
                #Asigna a cada lista su correspondiente campo
                campo_traduccion[nro_fila_char] = cadena_traduccion
                campo_titulo[nro_fila_char]     = cadena_titulo
                campo_link[nro_fila_char]       = cadena_link
                #Valida si se superó el tamaño del corpus                                
                if len(campo_traduccion) >= max_corpus_size:
                    #print('¡Error se superó el max_corpus_size!')
                    encontro_error = 1
                    break
                nro_fila = nro_fila + 1
        except ValueError:
            #print('¡Error en proceso de lectura de archivo csv!, fila: ' + str(nro_fila))
            encontro_error = 2

        num_leidos = num_leidos + 1

if encontro_error == 0:
    #Asigna ids y descripciones de cada campo
    ids_traduccion = list(campo_traduccion.keys())
    desc_traduccion = [campo_traduccion[qid] for qid in ids_traduccion]
    ids_titulo = list(campo_titulo.keys())
    desc_titulo = [campo_titulo[qid] for qid in ids_titulo]
    ids_link = list(campo_link.keys())
    desc_link = [campo_link[qid] for qid in ids_link]

    file = open("Resultado_1.txt", "w")
    file.write("***** Proceso de lectura de archivo terminado *****" + os.linesep)
    file.write("Nro. de registros leidos  :" + str(num_leidos) + os.linesep)
    file.write("Nro. de registros cargados: " + str(nro_fila-1))
    file.close()        
    
    #Crea Indice:
    if es.indices.exists(index="quora"):
        try:
            es_index = {
                "mappings": {
                  "properties": {
                    "field_traduccion": { "type": "text" },
                    "field_titulo": { "type": "keyword", "index": "false" },
                    "field_link": { "type": "keyword", "index": "false" },
                    "field_traduccion_vector": { "type": "dense_vector", "dims": 768 }
                  }
                }
            }

            es.indices.create(index='quora', body=es_index, ignore=[400])
            chunk_size = 50 #500
            print("Puedes detener la indexación presionando las teclas Ctrl+C):")
            with tqdm.tqdm(total=len(ids_traduccion)) as pbar:
                for start_idx in range(0, len(ids_traduccion), chunk_size):
                    end_idx = start_idx+chunk_size
                    embeddings = model.encode(desc_traduccion[start_idx:end_idx], show_progress_bar=False)
                    bulk_data = []
                    for ids, traduccion, titulo, link, embedding  in zip(ids_traduccion[start_idx:end_idx], desc_traduccion[start_idx:end_idx], desc_titulo[start_idx:end_idx], desc_link[start_idx:end_idx],embeddings):
                        bulk_data.append({
                                "_index": 'quora',
                                "_id": ids,
                                "_source": {
                                    "field_traduccion": traduccion,
                                    "field_titulo": titulo,
                                    "field_link": link,
                                    "field_traduccion_vector": embedding
                                }
                            })
                    helpers.bulk(es, bulk_data)
                    pbar.update(chunk_size)
                    
            file = open("Resultado_2.txt", "w")
            file.write("***** Proceso de generación de índice terminado correctamente *****")
            file.close()                    
            #print('Se generó el índice correctamente....')
        except:
            file = open("Resultado_2.txt", "w")
            file.write("¡Error durante el proceso de indexación!: ", sys.exc_info()[0], " Continue\n\n")
            file.close()                    
            #print("¡Error durante el proceso de indexaxión!: ", sys.exc_info()[0], " Continue\n\n")

elif encontro_error == 1:
    file = open("Resultado_1.txt", "w")
    file.write("¡Error en el Proceso de lectura de archivo csv, se superó el max_corpus_size!")
    file.close()
else:
    file = open("Resultado_1.txt", "w")
    file.write("¡Error en el proceso de lectura de archivo csv!, fila: " + str(nro_fila))
    file.close()


    
#if __name__ == '__indexar__':
#    run()
                
