# GPU-Accelerated-BERT-for-Fake-News-Detection-in-Filipino

### 📌 Project Overview
This project explores the parallelization of GPUs for optimizing a BERT-based fake news detection model in the Filipino language. Given the computationally intensive nature of transformer models, GPU acceleration was leveraged to improve training efficiency and model performance.

### 🚀 Key Highlights
- GPU Utilization: Used NVIDIA L4 GPU on Google Colab to parallelize training.
- Dataset: 1,603 Filipino news articles (fake and true) from Hugging Face.
- Model: BERT-based transformer fine-tuned for fake news classification.
- Performance Optimization:
  - Batch Size: 32 (optimal for speed vs. accuracy tradeoff)
  - Data Loaders: 2 workers for parallel data loading
  - Learning Rate: 0.00006
  - Epochs: 2 (to prevent overfitting)
- Results:
  - 94.39% accuracy
  - 76.26s per epoch (efficient training time)

### 🏭 Data Sources
The dataset consists of 1,603 news articles in Filipino:
- 801 labeled as fake news
- 802 labeled as true news
Dataset Features:
- text: The full news article in Filipino
- label:	0 = True News, 1 = Fake News
Text Length Distribution:
- Fake news articles have shorter text lengths on average compared to true news.
- The mean text length:
  - Fake News: ~121 words
  - True News: ~244 words

#### Word Cloud
Below are the most frequent words in fake and true news articles (excluding stop words):

<p align="center"> <img src="assets/wordcloud_fake.png" width="400"/> <img src="assets/wordcloud_true.png" width="400"/> </p>
Left: Fake News | Right: True News

#### Text Length Distribution
<p align="center"> <img src="assets/text_length_distribution.png" width="500"/> </p>
Fake news articles tend to be shorter than true news articles.

### 📊 Performance Analysis
#### 📌 Impact of Batch Size
Increasing batch size reduces training time but affects model accuracy:

<p align="center"> <img src="assets/batch_size_vs_accuracy.png" width="450"/> <img src="assets/batch_size_vs_time.png" width="450"/> </p>
Larger batch sizes reduce training time but may lower accuracy.

#### 📌 Speedup with Multiple Data Loaders
Using 2 data loaders optimized training efficiency without additional memory overhead:

<p align="center"> <img src="assets/dataloader_speedup.png" width="500"/> </p>
Beyond 2-3 data loaders, no significant speedup was observed.

#### 📌 Impact of Epochs
- Training for 2-3 epochs provided the best accuracy-speed tradeoff.
- More epochs resulted in overfitting, while fewer led to underfitting.

<p align="center"> <img src="assets/epochs_vs_accuracy.png" width="500"/> </p>
Accuracy improves up to 3 epochs, then plateaus.

## 🛠 Implementation Details
- Training Framework: PyTorch with Hugging Face’s Transformers library.
- Parallelization Techniques:
  - Used CUDA for GPU acceleration.
  - Tuned batch size and data loader workers for speedup.
  - Adjusted learning rate and epochs for convergence efficiency.

## 📜 Future Work
- Implement gradient accumulation & checkpointing for memory efficiency.
- Explore distributed training across multiple GPUs.
- Optimize inference speed for real-time deployment.

## 📎 References
1. Dataset: Hugging Face - Fake News Filipino
2. NVIDIA L4 GPU: NVIDIA Docs
3. Training Optimization: Hugging Face Performance Guide

