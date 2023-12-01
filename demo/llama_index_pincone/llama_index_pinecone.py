import os
import logging
import sys
import requests
import pinecone
from pathlib import Path

from llama_index import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
)

from llama_index.vector_stores import PineconeVectorStore
from llama_index.llms import OpenAI
from llama_index.storage.storage_context import StorageContext
from llama_index.indices.composability.graph import ComposableGraph
from llama_index.query_engine import CitationQueryEngine


os.environ["OPENAI_API_KEY"] = ""
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

wiki_titles = [
    "Toronto",
    "Seattle",
    "San Francisco",
    "Chicago",
    "Boston",
    "Washington, D.C.",
    "Cambridge, Massachusetts",
    "Houston",
]
pinecone_titles = [
    "toronto",
    "seattle",
    "san-francisco",
    "chicago",
    "boston",
    "dc",
    "cambridge",
    "houston",
]

for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            # 'exintro': True,
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    data_path = Path("data")
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()

api_key = ""
environment = "eu-west1-gcp"
index_name = "quickstart"

os.environ["PINECONE_API_KEY"] = api_key

llm = OpenAI(temperature=0, model="chatglm3-6b")
service_context = ServiceContext.from_defaults(llm=llm)

# Build city document index
city_indices = {}
for pinecone_title, wiki_title in zip(pinecone_titles, wiki_titles):
    metadata_filters = {"wiki_title": wiki_title}
    vector_store = PineconeVectorStore(
        index_name=index_name,
        environment=environment,
        metadata_filters=metadata_filters,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    city_indices[wiki_title] = VectorStoreIndex.from_documents(
        city_docs[wiki_title],
        storage_context=storage_context,
        service_context=service_context,
    )
    # set summary text for city
    city_indices[wiki_title].index_struct.index_id = pinecone_title

# set summaries for each city
index_summaries = {}
for wiki_title in wiki_titles:
    # set summary text for city
    index_summaries[wiki_title] = f"Wikipedia articles about {wiki_title}"

graph = ComposableGraph.from_indices(
    SimpleKeywordTableIndex,
    [index for _, index in city_indices.items()],
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk=50,
)

custom_query_engines = {
    graph.root_id: graph.root_index.as_query_engine(
        retriever_mode="simple", service_context=service_context
    )
}

query_graph_engine = graph.as_query_engine(
    custom_query_engines=custom_query_engines,
)

query_str = "Tell me more about Boston"
response_chat = query_graph_engine.query(query_str)

documents = SimpleDirectoryReader("./data/").load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    # 此处我们可以控制引用来源的粒度，默认值为512
    citation_chunk_size=512,
)

response = query_engine.query("Does Seattle or Houston have a bigger airport?")
print(response)
for source in response.source_nodes:
    print(source.node.get_text())

