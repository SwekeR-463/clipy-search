# install required libraries
# run commmand in terminal
# pip install datasets faiss-gpu transformers torch torchvision matplotlib

from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import faiss
import torch
import numpy as np
import matplotlib.pyplot as plt

# loading the dataset
fashion_ds = load_dataset("ashraq/fashion-product-images-small", split="train[:10%]")
# why 10%? coz there was cuda out of memory in my device
# fashion_ds

# helper func to visualize an image
def show_image(sample):
    img = sample["image"]
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{sample['productDisplayName']} ({sample['baseColour']})")
    plt.show()

# show_image(fashion_ds[0])  # display the first image

# loading the CLIP model 
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# embed the images
def encode_images(dataset):
    image_embeddings = []
    for item in dataset:
        image = clip_processor(images=item["image"], return_tensors="pt").to(device)
        with torch.no_grad():
            img_emb = clip_model.get_image_features(**image)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)  # normalize
        image_embeddings.append(img_emb.cpu().numpy())
    return np.vstack(image_embeddings)

image_embeddings = encode_images(fashion_ds)

# embed text metadata
fashion_texts = [
    f"{item['productDisplayName']} {item['articleType']} {item['baseColour']} "
    f"{item['gender']} {item['masterCategory']} {item['subCategory']} {item['usage']}"
    for item in fashion_ds
]
def encode_texts(texts):
    text_inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**text_inputs)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)  # normalize
    return text_emb.cpu().numpy()

text_embeddings = encode_texts(fashion_texts)

# build FAISS Index
# concatenate image and text embeddings for hybrid search
hybrid_embeddings = np.hstack([image_embeddings, text_embeddings]).astype("float32")
embedding_dim = hybrid_embeddings.shape[1]

# normalize hybrid embeddings
faiss.normalize_L2(hybrid_embeddings)

# build the index
index = faiss.IndexFlatL2(embedding_dim)
index.add(hybrid_embeddings)

# search queries
def search_fashion(query, k=5, mode="text"):
    if mode == "text":
        query_embedding = encode_texts([query])
    elif mode == "image":
        query_image = clip_processor(images=query, return_tensors="pt").to(device)
        with torch.no_grad():
            query_embedding = clip_model.get_image_features(**query_image)
            query_embedding /= query_embedding.norm(dim=-1, keepdim=True)
        query_embedding = query_embedding.cpu().numpy()
    else:
        raise ValueError("Invalid mode. Use 'text' or 'image'.")

    # duplicate the query embedding for hybrid search
    query_hybrid = np.hstack([query_embedding, query_embedding]).astype("float32")
    faiss.normalize_L2(query_hybrid)

    distances, indices = index.search(query_hybrid, k)
    print(f"Query: {query}\n")
    for i in range(k):
        sample = fashion_ds[indices[0][i]]
        print(f"Result {i+1}:")
        print(f"  Product Display Name: {sample['productDisplayName']}")
        print(f"  Article Type: {sample['articleType']}")
        print(f"  Color: {sample['baseColour']}")
        print(f"  Distance: {distances[0][i]:.4f}")
        show_image(sample)

# example text query
search_fashion("red shoes for men", k=3, mode="text")

# example image query 
# from PIL import Image
# query_image = Image.open("path_to_query_image.jpg")
# search_fashion(query_image, k=3, mode="image")