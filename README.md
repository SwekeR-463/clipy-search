# clipy-search

This project demonstrates a hybrid search engine for fashion products, enabling seamless querying using both text and images. Built using CLIP (Contrastive Language–Image Pretraining) and FAISS (Facebook AI Similarity Search), this system is tailored for e-commerce use cases, where users may want to search for products by describing them in text (e.g., "red sneakers for men") or by uploading a reference image.

The Fashion Product Images Small Dataset from Hugging Face forms the foundation of this project. It contains over 44,000 product images alongside rich metadata such as category, color, gender, and product names. The hybrid search engine combines the semantic power of CLIP embeddings with the scalability of FAISS indexing to deliver fast and accurate search results.

### Why Hybrid Search?
Traditional search engines rely heavily on keywords or exact matches, which can fail when users provide vague descriptions or visual references. A hybrid search engine:

* Understands Semantics: Matches products based on their meaning rather than just keywords or 
  pixel similarity.
* Supports Multimodal Queries: Enables searches using both text and image inputs.
* Efficiently Scales: Handles large datasets with FAISS for quick nearest-neighbor retrieval.
---

### Outputs

![Screenshot 2025-01-04 103606](https://github.com/user-attachments/assets/597d79b8-1c7c-4ebb-9757-cf29bfdb705b)

![Screenshot 2025-01-04 103659](https://github.com/user-attachments/assets/90c6b4eb-ebb8-47b3-b56f-083a4e206154)
