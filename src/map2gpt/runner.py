
import openai 
import tiktoken

from tqdm import tqdm

from pytube import YouTube
from typing import Optional, List, Dict, Any, Union

from map2gpt.log import logger 
from map2gpt.model import ExtractedFeatures

from map2gpt.strategies import merge_sentences_into_chunks, chunk_embeddings, gpt3_question_answering, load_transformer_model, gpt3_questions_generation
from map2gpt.strategies import convert_pdf_file2text, convert_wikipedia_page2text, convert_youtube_link2text, convert_text_into_sentences

from qdrant_client import QdrantClient, models

class GPTRunner:
    def __init__(self, openai_api_key:str, transformers_cache:str, transformer_model_name:str, qdrant:QdrantClient, device:str='cpu', tokenizer:str='gpt-3.5-turbo'):
        self.openai_api_key = openai_api_key
        self.transformer_model_name = transformer_model_name
        self.transformers_cache = transformers_cache
        self.device = device
        self.tokenizer = tiktoken.encoding_for_model(tokenizer)

        self.transformer_model = load_transformer_model(
            self.transformer_model_name,
            self.transformers_cache
        )

        openai.api_key = self.openai_api_key
        # initialze qdrant vector database 
        self.qdrant = qdrant
        self.vector_dimension = self.transformer_model.get_sentence_embedding_dimension()
        
    def create_qdrant_index(self, knowledge_base:List[ExtractedFeatures]):
        if len(knowledge_base) == 0:
            logger.warning('can not create qdrant index => no extracted features were found')
            return
        
        point_id = 0
        collection_created = False
        for extracted_features in tqdm(knowledge_base, desc='Creating Qdrant index'):
            if not collection_created:
                self.qdrant.recreate_collection(
                    collection_name=extracted_features.name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dimension,
                        distance=models.Distance.COSINE,
                    ),
                )
                collection_created = True

            zipped_chunks_embeddings = list(zip(extracted_features.chunks, extracted_features.embeddings))
            points:List[models.PointStruct] = []
            for chunk, embedding in zipped_chunks_embeddings:
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            'document_id': extracted_features.document_id,
                            'chunk': chunk,
                        }
                    )
                )
                point_id += 1

            self.qdrant.upsert(collection_name=extracted_features.name, points=points)

    def build_index_from_wikipedia_pages(self, urls:List[str], chunk_size:int, batch_size:int, name:str) -> List[ExtractedFeatures]:
        knowledge_base:List[ExtractedFeatures] = []
        for doc_id, url in enumerate(urls):
            extracted_sentences = convert_wikipedia_page2text(url)
            if extracted_sentences is None:
                continue
            logger.info(f'text was extracted from wikipedia {url}')
            chunks = merge_sentences_into_chunks(extracted_sentences, tokenizer=self.tokenizer, chunk_size=chunk_size)
            logger.info(f'nb_sentences : {len(extracted_sentences)} | nb_chunks : {len(chunks)}')
            
            embeddings = chunk_embeddings(
                transformer_model=self.transformer_model,
                chunks=chunks,  
                batch_size=batch_size,
                device=self.device
            ).tolist()

            extracted_features = {'document_id': doc_id, 'chunks': chunks, 'embeddings': embeddings, 'name':name}
            knowledge_base.append(ExtractedFeatures(**extracted_features))
        return knowledge_base
    
    def build_index_from_youtube_links(self, urls:List[str], chunk_size:int, batch_size:int, name:str, language:str) -> List[ExtractedFeatures]:
        knowledge_base:List[ExtractedFeatures] = []
        for doc_id, url in enumerate(urls):
            extracted_sentences = convert_youtube_link2text(url, language)
            if extracted_sentences is None:
                continue
            logger.info(f'text was extracted from youtube link {url}')
            chunks = merge_sentences_into_chunks(extracted_sentences, tokenizer=self.tokenizer, chunk_size=chunk_size)
            logger.info(f'nb_sentences : {len(extracted_sentences)} | nb_chunks : {len(chunks)}')
            
            embeddings = chunk_embeddings(
                transformer_model=self.transformer_model,
                chunks=chunks,  
                batch_size=batch_size,
                device=self.device
            ).tolist()

            extracted_features = {'document_id': doc_id, 'chunks': chunks, 'embeddings': embeddings, 'name':name}
            knowledge_base.append(ExtractedFeatures(**extracted_features))
        return knowledge_base

    def build_index_from_pdf_files(self, path2pdf_files:List[str], chunk_size:int, batch_size:int, name:str) -> List[ExtractedFeatures]:
        knowledge_base:List[ExtractedFeatures] = []
        for doc_id, path2pdf_file in enumerate(path2pdf_files):
            extracted_sentences = convert_pdf_file2text(path2pdf_file)
            if extracted_sentences is None:
                continue
            logger.info(f'text was extracted from pdf file {path2pdf_file}')
            chunks = merge_sentences_into_chunks(extracted_sentences, tokenizer=self.tokenizer, chunk_size=chunk_size)
            logger.info(f'nb_sentences : {len(extracted_sentences)} | nb_chunks : {len(chunks)}')

            embeddings = chunk_embeddings(
                transformer_model=self.transformer_model,
                chunks=chunks,  
                batch_size=batch_size,
                device=self.device
            ).tolist()

            extracted_features = {'document_id': doc_id, 'chunks': chunks, 'embeddings': embeddings, 'name':name}
            knowledge_base.append(ExtractedFeatures(**extracted_features))
        return knowledge_base

    def build_index_from_texts(self, texts:List[str], chunk_size:int, batch_size:int, name:str) -> List[ExtractedFeatures]:
        knowledge_base:List[ExtractedFeatures] = []
        for doc_id, text in enumerate(texts):
            extracted_sentences = convert_text_into_sentences(text)
            if extracted_sentences is None:
                continue
            chunks = merge_sentences_into_chunks(extracted_sentences, tokenizer=self.tokenizer, chunk_size=chunk_size)
            logger.info(f'nb_sentences : {len(extracted_sentences)} | nb_chunks : {len(chunks)}')

            embeddings = chunk_embeddings(
                transformer_model=self.transformer_model,
                chunks=chunks,  
                batch_size=batch_size,
                device=self.device
            ).tolist()

            extracted_features = {'document_id': doc_id, 'chunks': chunks, 'embeddings': embeddings, 'name':name}
            knowledge_base.append(ExtractedFeatures(**extracted_features))
        return knowledge_base


    def query_index(self, query:str, collection_name:str, description:str, top_k:int=7, source_k:int=2) -> Optional[str]:
        assert top_k >= 1, 'top_k must be greater than or equal to 1'
        assert source_k <= top_k, 'source_k must be less than or equal to top_k'
        
        query_embedding = self.transformer_model.encode(query, device=self.device, show_progress_bar=False)
        qdrant_response = self.qdrant.search(collection_name=collection_name, query_vector=query_embedding, limit=top_k)
                
        chunks_acc = []
        for hit in qdrant_response:
            chunk = hit.payload['chunk']
            chunks_acc.append(chunk)
        corpus_context = '\n'.join(chunks_acc)

        try:
            completion_response = gpt3_question_answering(
                query=query,
                name=collection_name,
                description=description,
                corpus_context=corpus_context,
            )
            return completion_response
        
        except Exception as e:
            logger.error(e)
        return None

    def generate_questions(self, query:str, collection_name:str, top_k:int=7) -> Optional[str]:
        assert top_k >= 1, 'top_k must be greater than or equal to 1'
        query_embedding = self.transformer_model.encode(query, device=self.device, show_progress_bar=False)
        qdrant_response = self.qdrant.search(collection_name=collection_name, query_vector=query_embedding, limit=top_k)
                
        chunks_acc = []
        for hit in qdrant_response:
            chunk = hit.payload['chunk']
            chunks_acc.append(chunk)
        corpus_context = '\n'.join(chunks_acc)

        try:
            completion_response = gpt3_questions_generation(corpus_context)
            return completion_response
        except Exception as e:
            logger.error(e)
        return None
    