from datasets import load_dataset
import pandas
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm # to show progress bar
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Cohere
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from config import OPENAI_API_KEY, PINECONE_API_KEY, COHERE_API_KEY
from model import SentenceTransformerEmbeddings

model = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base")
custom_embeddings = SentenceTransformerEmbeddings(model)

# data = load_dataset('squad', split='train')
# df = data.to_pandas()
# print(df.head(1))
# print(df.iloc[0]['question'])
# remove duplicates from the dataframe with same contexts
# df.drop_duplicates(subset='context', keep='first', inplace=True)

# db dimension 768
pc = Pinecone(api_key=PINECONE_API_KEY)

# pc.create_index(
#     name="ai-agent1",
#     dimension=768,
#     metric="dotproduct",
#     spec=ServerlessSpec(    
#         cloud="aws",
#         region="us-east-1"
#     )
# )

index = pc.Index('ai-agent')

# df_sample = df.sample(20, random_state=45)
# batch_size = 10

# for i in range(0, len(df_sample), batch_size):
#     i_end = min(i+batch_size,len(df_sample))
#     batch = df_sample.iloc[i:i_end].copy()
    
#     meta_data = [{'title': row['title'], 'context': row['context']} for i,row in batch.iterrows()]
    
#     docs = batch['context'].tolist() # pd.series to python list
#     emb_vectors = custom_embeddings.embed_documents(docs)
#     ids = batch['id'].tolist()

#     # upsert
#     to_upsert = zip(ids, emb_vectors, meta_data)
#     index.upsert(vectors=to_upsert)

vectorstore = PineconeVectorStore(index,custom_embeddings, "context", pinecone_api_key=PINECONE_API_KEY)
# query = "what is physical data base design?"
# res = vectorstore.similarity_search(query, k=3)
# print(res)

llm = Cohere(cohere_api_key=COHERE_API_KEY)

conv_mem = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

qa = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = 'stuff',
    retriever = vectorstore.as_retriever()
)

# query = "what is physical data base design?"
query = "who won the Chess World Cup in 2009?"

# print(qa.invoke(query))

tools = [
    Tool(
        name='Knowledge Base',
        func=qa.invoke,
        description=('use this when answering based on knowledge')
    )
]

agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
    max_iteration=3,
    early_stopping_method='generate',
    memory=conv_mem
)

print(agent("who won the Chess World Cup in 2009?"))