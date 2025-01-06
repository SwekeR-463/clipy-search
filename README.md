# clipy-search

This project demonstrates a hybrid search engine for fashion products, enabling seamless querying using both text and images. Built using CLIP (Contrastive Language–Image Pretraining) and FAISS (Facebook AI Similarity Search), this system is tailored for e-commerce use cases, where users may want to search for products by describing them in text (e.g., "red sneakers for men") or by uploading a reference image.

---

### Dataset
The Fashion Product Images Small Dataset from [Hugging Face](https://huggingface.co/datasets/ashraq/fashion-product-images-small) forms the foundation of this project. It contains over 44,000 product images alongside rich metadata such as category, color, gender, and product names. The hybrid search engine combines the semantic power of CLIP embeddings with the scalability of FAISS indexing to deliver fast and accurate search results.

---

### Why Hybrid Search?
Traditional search engines rely heavily on keywords or exact matches, which can fail when users provide vague descriptions or visual references. A hybrid search engine:

* Understands Semantics: Matches products based on their meaning rather than just keywords or 
  pixel similarity.
* Supports Multimodal Queries: Enables searches using both text and image inputs.
* Efficiently Scales: Handles large datasets with FAISS for quick nearest-neighbor retrieval.

---

### How I did this?

1. **Dataset Loading and Visualization**

* The dataset is loaded from Hugging Face’s ashraq/fashion-product-images-small. It contains over 44,000 product images with metadata like productDisplayName, baseColour, gender, and more.
* A helper function, show_image, is used to display the images with metadata for visualization.

2. **Embedding with CLIP**

* CLIP Model: The pre-trained CLIP model (openai/clip-vit-base-patch32) is used to embed both text and images into a shared latent space.
* Image Embeddings: Each image is processed using the CLIP image encoder, producing a semantic representation of its content.
* Text Embeddings: Metadata such as productDisplayName, articleType, and baseColour is concatenated and passed through the CLIP text encoder.

3. **Hybrid Embedding**

* The text and image embeddings are concatenated into hybrid embeddings, combining visual and semantic information for each product.
* These embeddings are normalized to ensure compatibility with FAISS similarity search.

4. **Building the FAISS Index**

* FAISS Index: The IndexFlatL2 index is used to store the hybrid embeddings. It supports efficient similarity search using L2 distance.
* The index is populated with the normalized hybrid embeddings, allowing for scalable and fast nearest-neighbor retrieval.
5. **Search Functionality**

* Text Query: Converts user input (e.g., "red shoes for men") into an embedding using the CLIP text encoder.
* Image Query: Embeds the uploaded image using the CLIP image encoder.
* Hybrid Search: The query embedding is matched against the hybrid embeddings in the FAISS index, returning the top-k nearest neighbors.

6. **Results Visualization**

* The search results include metadata (e.g., productDisplayName, baseColour) and a visual representation of the retrieved product images.

---

### Outputs

![Screenshot 2025-01-04 103606](https://github.com/user-attachments/assets/597d79b8-1c7c-4ebb-9757-cf29bfdb705b)

![Screenshot 2025-01-04 103659](https://github.com/user-attachments/assets/90c6b4eb-ebb8-47b3-b56f-083a4e206154)

### Setup & Usage

#### Setup

1. Clone the Repository
```python
git clone https://github.com/SwekeR-463/clipy-search.git
cd clipy-search
```

2. Install Dependencies
Create a virtual environment and install the required libraries:
```python
python -m venv env
source env/bin/activate # for linux/macOS
env\Scripts\activate # for windows
pip install -r requirements.txt
```

3. Download Pretrained CLIP Model The project uses OpenAI's `clip-vit-base-patch32`. It will automatically download when the code is executed.

4. GPU Support Ensure that PyTorch is installed with CUDA support to leverage GPU acceleration. You can check if PyTorch detects your GPU:
```python
import torch
print(torch.cuda.is_available())
```

---

#### Usage

Once the setup is complete, you can run the script to perform hybrid search.

1. Run the Script
```python
python app.py
```

2. Peform Searches

* Text Query: Enter a text description, such as:
```python
query = "red tshirts for men"
search_fashion(query, k=5, mode="text")
```

* Image Query: Provide an image (e.g., query_image.jpg) and run:
```python
from PIL import Image
query_image = Image.open("query_image.jpg")
search_fashion(query_image, k=5, mode="image")
```
The system will retrieve products visually similar to the input image.

3. Results Visualization The search results include:
* Product metadata like productDisplayName and baseColour.
* Images of the retrieved products.

---
