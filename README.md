# Snippets


## Least Recently Used (LRU) Cache

**lru_cache.py**

Least Recently Used Cache stores key-value pairs up to a capacity, after which it removes the least recently accessed entries to accommodate additional entries. This implementation utilizes a doubly-linked list and dictionary. The doubly-linked list orders the data formatted as nodes, allowing add and remove in constant O(1) time. The dictionary stores nodes as values, enabling constant O(1) time access to a node when given the key.

**lru_cache_test.py**

This contains unit tests for the LRU cache.


## Binary Tree Serializer

**binary_tree_serializer.py**

Binary Tree Serializer converts a binary tree to a string, then reconstructs the original binary tree using that string. This is done in O(n) time because it requires a single traversal across all nodes for either representation.

**binary_tree_serializer_test.py**

This contains unit tests for the binary tree serializer, and a function evaluating whether two binary trees are equivalent.


## Multimodal Ophthalmoscopic Image Fusion Using Paired Autoencoders
_Complete code can be found in the **Paired-Autoencoder-Image-Fusion** repository._

**autoencoder_datasets.py**

These custom Datasets are inherited or reworked from PyTorch Datasets. The Datasets store training or testing, single-image-type or matched fundus and FLIO data, for efficient batched and multiprocessing inputs into the machine learning model via PyTorch DataLoader.

**paired_autoencoder_traintest_functions.py**

The train and test functions run training and testing for matched fundus-FLIO data on the paired Autoencoder models with jointly constrained latent spaces. The additional functions calculate individual and total losses given loss criterion, contributing to the changes of the models' weights and thus the learning process.

### Background

Age-related macular degeneration (AMD) is a progressive retinal condition characterized by the presence of drusen, accumulated deposits between the ocular membranes. Current AMD studies largely rely on RGB fundus photography to identify visible drusen as an indicator of disease progression.

Fluorescence lifetime imaging ophthalmoscopy (FLIO) is a novel data source that records changes in fundus autofluorescence, detecting the presence of AMD-related biochemical processes. FLIO data analysis has the potential to identify early stages of AMD development.

### Goals

1) Preprocess fundus and FLIO data into appropriate formats and similar sizes
2) Create paired autoencoder model with constrained feature spaces to take data from these two sources; output single image capturing information from both sources
3) Test different model architectures and loss functions to optimize results
