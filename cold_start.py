import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import clip
import torch
from PIL import Image
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import List

class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="ViT-L/14"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow, input is a list of tuples containing img path and text data
        images = []
        texts = []

        for image_path, text in input:
            image = self.preprocess(Image.open(image_path).convert("RGB"))
            images.append(image)
            texts.append(text)

        # Stack all images into a single batch tensor and move to device
        images_tensor = torch.stack(images).to(self.device)

        # Tokenize all texts as a batch and move to device
        text_tokens = clip.tokenize(texts).to(self.device)

        with torch.no_grad():
            # Encode all images and normalize
            image_features = self.model.encode_image(images_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Encode all texts and normalize
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Combine image and text features and re-normalize
            combined = (image_features + text_features) / 2
            combined = combined / combined.norm(dim=-1, keepdim=True)

        # Convert to list of vectors on CPU
        return combined.cpu().tolist()

class VectorStore():
    def __init__(self):
        self.collection = None
        self.client = None
        self.embedding_function = None


    def connect(self, database_name, database_path, my_embedding_fn):
        """
        connects to existing database
        """
        self.client = chromadb.PersistentClient(path=database_path)
        self.collection = self.client.get_collection(name=database_name, embedding_function=my_embedding_fn)
        self.embedding_function = my_embedding_fn
        return self.collection

    def create_db(self, database_name,database_path, embedding_function):
        """
        creates a persistent database
        """
        self.client = chromadb.PersistentClient(path=database_path)
        
        self.collection = self.client.create_collection(
            name=database_name, 
            embedding_function=embedding_function,
            metadata={
                "description": "collection containing steam game screenshots and tags",
                "created": str(datetime.now())
            }  
        )
        self.embedding_function = embedding_function
        return self.collection
    
    def add_data(self, documents: list, ids: list):

        embeddings = self.embedding_function(documents)

        self.collection.add(
            embeddings=embeddings,
            ids=ids
        )

    def get_embeddings(self, documents):

        embeddings = self.embedding_function(documents)

        return embeddings
    
    def recommend(self, documents):

        embeddings = self.get_embeddings(documents)

        combined_embeddings = self.combine_and_normalize(embeddings)

        ids = self.collection.query(

            query_embeddings=[combined_embeddings]
        )    

        return ids
    
    def combine_and_normalize(self, embeddings: List[List[float]]):
        # Convert to tensor
        tensor = torch.tensor(embeddings)  # shape: [B, D]

        # Combine (e.g., average across B embeddings)
        combined = tensor.mean(dim=0)  # shape: [D]

        # Normalize (L2 norm)
        normalized = combined / combined.norm(p=2)

        return normalized.tolist()

# df_train = pd.read_csv('recommender_training_data.csv')
# app_ids = [str(app_id) for app_id in df_train['app_id'].unique().tolist()]

# tags_df = pd.read_csv("steam_game_description_376110.csv")
# tags_map = {}
# for index, row in tags_df.iterrows():
#     app_id = str(row["app_id"])
#     tag = row["tag"]
#     tags_map[app_id] = "None" if pd.isna(tag) else tag

# folder = Path('steam_images_store')
# img_names = set(f.name[:-4] for f in folder.iterdir() if f.is_file())

# documents = []
# ids = []
# for app_id in app_ids:
#     if app_id in tags_map and app_id in img_names:
#         img_path = "steam_images_store/" + app_id + ".jpg"
#         text_data = tags_map[app_id]
#         documents.append((img_path, text_data))
#         ids.append(app_id)

# # Combine into rows
# rows = []
# for app_id, (img_path, text) in zip(ids, documents):
#     rows.append({
#         "app_id": app_id,
#         "img_path": img_path,
#         "text": text
#     })

# # Create DataFrame
# df = pd.DataFrame(rows)

# # Save to CSV
# df.to_csv("app_image_text_data.csv", index=False)

# emb_function = MyEmbeddingFunction()
# vector_store = VectorStore()
# vector_store.create_db("steam_vector_db", "steam_db", emb_function)

# batch_size = 64
# total = len(documents)

# for i in range(0, len(documents), batch_size):
#     batch_docs = documents[i:i+batch_size]
#     batch_ids = ids[i:i+batch_size]
#     vector_store.add_data(
#         documents=batch_docs,
#         ids = batch_ids
#     )
#     print(f"Batch {i}")

def retrieve_coldstart_recommendations(documents):
    emb_function = MyEmbeddingFunction()
    vector_store = VectorStore()
    vector_store.connect("steam_vector_db", "steam_db", emb_function)

    print(vector_store.collection.count())

    print(vector_store.embedding_function)

    recommendations = vector_store.recommend(documents)
    print(recommendations)

    return recommendations['ids'][0]