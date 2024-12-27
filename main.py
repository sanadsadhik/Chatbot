from datasets import load_dataset
import pandas
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm # to show progress bar

from config import OPENAI_API_KEY, PINECONE_API_KEY

def get_embedding(text):
    text = text.replace('\n', ' ')
    return retriever.encode(text).tolist()

# openai_api_key = OPENAI_API_KEY
retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base")

data = load_dataset('squad', split='train')
df = data.to_pandas()
# print(df.head(1))
# print(df.iloc[0]['question'])
# remove duplicates from the dataframe with same contexts
df.drop_duplicates(subset='context', keep='first', inplace=True)

# client = OpenAI(api_key=openai_api_key)
# MODEL = "text-embedding-ada-002"
# response = client.embeddings.create(
#     input='i love openai',
#     model=MODEL
# )
# print(response)

# vec = get_embedding('I am trying a new text \n And see what happend')
# print(f'required vector dim: {vec}')

# db dimension 768
# pinecone_api_key = PINECONE_API_KEY
# pc = Pinecone(api_key=PINECONE_API_KEY)

# pc.create_index(
#     name="ai-agent",
#     dimension=768,
#     metric="dotproduct",
#     spec=ServerlessSpec(    
#         cloud="aws",
#         region="us-east-1"
#     )
# )

# index = pc.Index('ai-agent')

df_sample = df.sample(20, random_state=45)
batch_size = 10

for i in range(0, len(df_sample), batch_size):
    i_end = min(i+batch_size,len(df_sample))
    batch = df_sample.iloc[i:i_end].copy()
    meta_data = [{'title': row['title'], 'context': row['context']} for i,row in batch.iterrows()]

