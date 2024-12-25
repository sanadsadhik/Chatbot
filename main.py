from datasets import load_dataset
import pandas
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from config import OPENAI_API_KEY, PINECONE_API_KEY

# openai_api_key = OPENAI_API_KEY
retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base")

# data = load_dataset('squad', split='train')

# df = data.to_pandas()
# print(df.head(1))
# print(df.iloc[0]['question'])
# remove duplicates from the dataframe with same contexts
# df.drop_duplicates(subset='context', keep='first', inplace=True)

# client = OpenAI(api_key=openai_api_key)
# MODEL = "text-embedding-ada-002"
# response = client.embeddings.create(
#     input='i love openai',
#     model=MODEL
# )
# print(response)

def get_emedding(text):
    text = text.replace('\n', ' ')
    return retriever.encode(text).tolist()

vec = get_emedding('I am trying a new text \n And see what happend')
print(f'required vector dim: {vec}')

# db dimension 768
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)

pc.create_index(
    name="ai-agent",
    dimension=768,
    metric="dotproduct",
    spec=ServerlessSpec(    
        cloud="aws",
        region="us-east-1"
    )
)

index = pc.Index('ai-agent')

