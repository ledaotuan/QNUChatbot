import requests
import openai 
import pandas as pd
from configs import *
import json
import numpy as np
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
)
from llama_index.vector_stores import PineconeVectorStore
from llama_index.llms import OpenAI
import pinecone
import os 
from langchain.embeddings.cohere import CohereEmbeddings
from llama_index.storage.storage_context import StorageContext
from llama_index.schema import MetadataMode
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    EntityExtractor,
    MetadataFeatureExtractor,
)
from llama_index.text_splitter import TokenTextSplitter
from llama_index.storage.storage_context import StorageContext
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index import SimpleDirectoryReader
from llama_index.llms import ChatMessage, MessageRole
from llama_index.prompts import ChatPromptTemplate


os.environ["PINECONE_API_KEY"] = "cc2b8f47-271c-4e8c-a355-28a4189cc958"
os.environ["COHERE_API_KEY"] = "rKtcRh4acfK3AVixyQtO9fX8aop4nxbgziWkheXB"
os.environ["OPENAI_API_KEY"] = "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"
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")
embeddings = CohereEmbeddings(model = "multilingual-22-12")
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embeddings)

text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)
metadata_extractor = MetadataExtractor(
    extractors=[
        TitleExtractor(nodes=5, llm=llm),
        QuestionsAnsweredExtractor(questions=3, llm=llm),
        EntityExtractor(prediction_threshold=0.5),
        KeywordExtractor(keywords=10, llm=llm),
    ],
)

node_parser = SimpleNodeParser.from_defaults(
    text_splitter=text_splitter,
    metadata_extractor=metadata_extractor,
)



index_name = "data" 
pinecone_env = 'gcp-starter'
pinecone.init(environment=pinecone_env)

#pinecone.delete_index(index_name)

if index_name in pinecone.list_indexes():
    vector_store = PineconeVectorStore(pinecone.Index(index_name))
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)
    print("nhanh")
else:
    pinecone.create_index(
      name=index_name,
      pod_type='p1.x1',
      metric='cosine',
      dimension=768  # The Cohere embedding model `multilingual-22-12' uses 768 dimensions  
    )
    vector_store = PineconeVectorStore(
            index_name=index_name,
            environment=pinecone_env,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store) 
    docs = SimpleDirectoryReader(input_files = [r"C:\Users\ADMIN\Desktop\NCKH_2023\thesis-main\thesis-main\cleaned_data\cong-nghe-ky-thuat-hoa-hoc-3904.txt"]).load_data()
    tud_node = node_parser.get_nodes_from_documents(docs)
    
    for i in range(len(tud_node)):
        tud_node[i].metadata["entities"] = ", ".join(tud_node[i].metadata["entities"])
    index = VectorStoreIndex(
        nodes = tud_node,
        storage_context=storage_context,
        service_context=service_context,
        show_progress = True
    )
    print("Cham")

chat_text_qa_msgs = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "Bạn là người tư vấn tuyển sinh của trường Đại Học Quy Nhơn.\n"
                "Luôn luôn trả lời câu hỏi sử dụng thông tin được cung cấp."

        )),
        ChatMessage(
            role=MessageRole.USER,
            content=(
                "Thông tin ngữ cảnh được cho như sau:\n"
                "---------------------\n"
                "{context_str}\n" 
                "---------------------\n"
                "Dựa trên ngữ cảnh và không dùng kiến thức biết trước, "
                "Trả lời câu hỏi: {query_str}\n"
            ),
        ),
    ]
text_qa_template = ChatPromptTemplate(chat_text_qa_msgs) # Thêm promt

engine = index.as_query_engine(similarity_top_k=3, retriever_mode="embedding", service_context=service_context,
                               text_qa_template=text_qa_template) 
final_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[
        QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(
                name="tud_documents",
                description="Cung cấp thông tin về ngành toán ứng dụng của trường Đại học Quy Nhơn",
            ),
        ),],
    service_context=service_context,
    use_async=True,
)