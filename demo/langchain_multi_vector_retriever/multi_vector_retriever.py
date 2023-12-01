import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.document import Document
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser

def func1():
    # 方法1 分割文档，生成更小的组块
    child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    sub_docs = []
    for i, doc in enumerate(docs):
        _id = doc_ids[i]
        _sub_docs = child_text_splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id
        sub_docs.extend(_sub_docs)
    return sub_docs

def func2():
    # 方法2 生成摘要
    chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
        | ChatOpenAI(max_retries=0)
        | StrOutputParser()
    )
    summaries = chain.batch(docs, {"max_concurrency": 5})
    summary_docs = [Document(page_content=s,metadata={id_key: doc_ids[i]}) for i, s in enumerate(summaries)]
    return summary_docs

def func3():
    # 方法3 生成假设性的问题
    functions = [
        {
        "name": "hypothetical_questions",
        "description": "Generate hypothetical questions",
        "parameters": {
            "type": "object",
            "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "string"
                },
            },
            },
            "required": ["questions"]
        }
        }
    ]
    chain = (
        {"doc": lambda x: x.page_content}
        # Only asking for 3 hypothetical questions, but this could be adjusted
        | ChatPromptTemplate.from_template("Generate a list of 3 hypothetical questions that the below document could be used to answer:\n\n{doc}")
        | ChatOpenAI(max_retries=0, model="gpt-4").bind(functions=functions, function_call={"name": "hypothetical_questions"})
        | JsonKeyOutputFunctionsParser(key_name="questions")
    )
    hypothetical_questions = chain.batch(docs, {"max_concurrency": 5})
    question_docs = []
    for i, question_list in enumerate(hypothetical_questions):
        question_docs.extend([Document(page_content=s,metadata={id_key: doc_ids[i]}) for s in question_list])
    return question_docs

def get_docs(func_num):
    if func_num==1:
        return func1()
    elif func_num==2:
        return func2()
    elif func_num==3:
        return func3()
    else:
        return []

# 检索过程的代码
loaders = [
    TextLoader('../../paul_graham_essay.txt'),
    TextLoader('../../state_of_the_union.txt'),
]
docs = []
for l in loaders:
    docs.extend(l.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
docs = text_splitter.split_documents(docs)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="full_documents",
    embedding_function=OpenAIEmbeddings()
)
# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

doc_ids = [str(uuid.uuid4()) for _ in docs]
# 根据不同方法类型，选择不同策略
func_num =1
candidate_docs = get_docs(func_num)

retriever.vectorstore.add_documents(candidate_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

# search
retriever.vectorstore.similarity_search("justice")[0]