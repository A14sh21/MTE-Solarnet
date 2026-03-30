# Solar Panel Defect Detection and Diagnostic System
## Introduction
The rapid expansion of solar energy infrastructure necessitates automated, highly accurate maintenance solutions. Micro-cracks, hot spots, bird drops, and dust accumulation can significantly degrade photovoltaic (PV) module efficiency. Traditional manual inspections are time-consuming and prone to human error. This project presents a state-of-the-art, end-to-end AI system that not only detects and classifies solar panel defects using an advanced computer vision pipeline but also generates contextual, actionable maintenance reports using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG).

## Objective
The primary objective of this project is to build a highly accurate, explainable, and user-friendly diagnostic tool for solar infrastructure. Specifically, the system aims to:

1. Achieve superior feature extraction and spatial awareness in identifying structural anomalies on PV modules.
2. Reduce false positives by focusing the model's computation on critical regions using advanced attention mechanisms.
3. Bridge the gap between raw classification outputs and actionable engineering insights by utilizing an LLM backed by a specialized knowledge base (RAG).
4. Provide a seamless, interactive web interface for end-users and maintenance personnel.

## Architecture
The system architecture is a hybrid design combining a deep learning vision module with a generative AI language module, orchestrated through a web framework.

1. Vision Backbone (EfficientNet-B4): Chosen for its optimal balance between parameter efficiency and high accuracy. It serves as the primary feature extractor, analyzing the complex textures and patterns of solar cells.

2. Spatial Enhancement (Coordinate Attention): Integrated into the EfficientNet-B4 architecture. Unlike standard channel attention, Coordinate Attention embeds positional information into channel maps, allowing the network to accurately localize the exact spatial coordinates of defects like micro-cracks or gridline interruptions.

3. Knowledge Retrieval (RAG - Retrieval-Augmented Generation): A vector database populated with solar panel maintenance manuals, defect resolution protocols, and industry standards.

4. Reasoning Engine (LLM): A Large Language Model integrated via API, tasked with interpreting the vision model's output and cross-referencing it with the RAG database.

5. User Interface (Gradio): A lightweight, robust web interface that handles image uploads, triggers the inference pipeline, and renders the visual and textual results.

## Methodology
The development of this system was executed in sequential phases:

1. Data Preprocessing and Augmentation: Solar panel electroluminescence (EL) or infrared (IR) images were standardized, resized, and subjected to varying augmentation techniques (rotation, contrast adjustment) to ensure model robustness against different environmental conditions.

2. Vision Model Training: The EfficientNet-B4 model was fine-tuned on the annotated dataset. The Coordinate Attention mechanism was injected to ensure the model learned to prioritize spatial features critical to defect localization rather than background noise.

3. Vector Database Construction: Technical documents regarding solar panel maintenance were chunked, converted into embeddings using a sentence-transformer model, and indexed in a vector database to enable rapid semantic search.

4. Pipeline Integration: The vision model's classification output (defect type and confidence score) was programmed to act as the query trigger for the RAG system.

5. Interface Deployment: The Gradio application was scripted to handle asynchronous requests, ensuring that the heavy computation of the vision model and the LLM generation occurred without locking the user interface.

## How the Model Works
The end-to-end inference process operates seamlessly when a user interacts with the system:

1. Input: The user uploads an image of a solar panel (e.g., drone footage or an EL scan) through the Gradio web interface.

2. Vision Processing: The image is passed to the EfficientNet-B4 + Coordinate Attention model. The network extracts features and precisely localizes anomalies, ultimately outputting a defect classification (e.g., "Hotspot detected") and a confidence metric.

3. Information Retrieval: The predicted defect class is formatted into a search query. The RAG system searches its vector database to retrieve the most relevant technical documentation and standard operating procedures for handling that specific defect.

4. Contextual Generation: The LLM receives a strict prompt containing the vision model's prediction, the confidence score, and the retrieved documents from the RAG system. The LLM synthesizes this information into a clear, professional diagnostic report detailing the nature of the defect, potential causes, and recommended mitigation steps.

5. Output: The Gradio interface updates to display the results to the user, providing both the technical classification and the comprehensive, easy-to-read LLM-generated maintenance report.
