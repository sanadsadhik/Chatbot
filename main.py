from datasets import load_dataset
import pandas
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm # to show progress bar
from langchain_pinecone import PineconeVectorStore

from config import OPENAI_API_KEY, PINECONE_API_KEY
from model import SentenceTransformerEmbeddings

model = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base")
custom_embeddings = SentenceTransformerEmbeddings(model)

data = load_dataset('squad', split='train')
df = data.to_pandas()
# print(df.head(1))
# print(df.iloc[0]['question'])
# remove duplicates from the dataframe with same contexts
df.drop_duplicates(subset='context', keep='first', inplace=True)

# db dimension 768
pinecone_api_key = PINECONE_API_KEY
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

index = pc.Index('ai-agent1')

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

query = "what is physical data base design?"

res = vectorstore.similarity_search(query, k=3)
print(res)