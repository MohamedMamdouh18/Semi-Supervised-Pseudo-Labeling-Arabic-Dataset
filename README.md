# Semi-Supervised-Pseudo-Labeling-Arabic-Dataset
This repository a Semi-Supervised approach to perform sentiment annotation for a large Arabic text corpus. Sentiment analysis is a crucial natural language processing (NLP) task, where the objective is to automatically determine the sentiment expressed in a piece of text, such as positive and negative.

**Note** The Notebook won't open in github so you can download it or preview it from [here](https://colab.research.google.com/drive/18opmnSi1hIbpUems0qG6NyTbRGsa-4o4?usp=sharing)

## Table of Contents

- [Motivation](#Motivation)
- [Dataset](#Dataset)
- [Installed Packages](#Installed-Packages)
- [Methodology](#Methodology)
    - [Preprocessing](#Preprocessing)
    - [Embeddings](#Embeddings)
    - [Tune Hyperparameters](#Tune-Hyperparameters)
    - [Proposed Architecture](#Proposed-Architecture)
- [Results & Discussion](#Results-&-Discussion)
- [Authors](#Authors)

## Motivation
  At a time when research in the field of sentiment analysis tends to study advanced topics in languages, such as English, other languages such as Arabic still suffer from basic problems and challenges, most notably the availability of large corpora. Furthermore, manual annotation is time-consuming and difficult when the corpus is too large. This Repository implements a semi-supervised self-learning technique, to extend an Arabic sentiment annotated corpus with unlabeled data and tries to make the whole dataset annotated.

## Dataset
[LABR: A Large Scale Arabic Book Reviews Dataset](https://aclanthology.org/P13-2088) (Aly & Atiya, ACL 2013) was used. The dataset contains 16,448 book reviews. The classes are positive and negative. 10,000 reviews only are used in training and seeding while the rest is considered unlabeled data and our main objective is to implement a method to label them.

## Installed Packages
The libraries that needed to be installed before beginning :
```
!pip install arabert
!pip install arabic_reshaper
!pip install python-bidi
!pip install camel-tools
!pip install gensim
!pip install emoji
!pip install transformers
!pip install sentencepiece
!pip install tokenizers
!pip install lime
```

## Methodology

### Preprocessing
We used multiple tools for multiple purposes :

- **Py-Arabic** handles removing tashkel and converting English numbers to Arabic.
- **Camel-tools** removes tatweel and normalize some Arabic letters eg: (ة🡪ه).
- **ARA-Bert Preprocessor**  removes tashkeel, seperate punctuations from texts, removes repeated letters in a word Moreover and removes emojis.
- **Customized Functions** removes English words, punctuations and Arabic stopwords.
 
### Embeddings

After tokenizing each review using white space tokenizer , ARAVEC model pretrained on twitter reviews() is used to vectorize each token to use it as input to the given model.

- #### Embeddings for ML models
  Our approach was to get the embedding of each word in the sentence then get their average so the input to the ML model was a vector with 300 dimensions.

- #### Embeddings for GRU model
  Using ARAVEC embeddings with non-trainable embedding layer yielded the same results as using trainable embedding layer so our final approach was to pass inputs through a **trainable** embedding layer to the gru model.

### Tune Hyperparameters

- For LR, we use RandomizedSearchCV to search over a large scale of parameters. This helps us to narrow down the search range and find a good set of parameters for GridSearchCV to use.
- For LinearSVM, we did the same but we used HalvingRandomSearchCV instead of RandomizedSearchCV.
- For GRU Model, it took much time using HalvingRandomSearchCV so we tried different architectures until we got the current one.

### Proposed Architecture

Our approach was based on [Refrence](https://doi.org/10.3390/app11052434)

![Capture](https://github.com/MohamedMamdouh18/Semi-Supervised-Pseudo-Labeling-Arabic-Dataset/assets/63814228/83d07cbd-4c4e-4652-9985-c2875432ebd6)

- Depending on one model to annotate a dataset even with relatively high accuracy(80%) makes it an infeasible solution as wrong annotations might lead to failure of any model that uses the dataset.
- Instead an Ensemble of 3 classifiers (Logistic Regression, Linear SVM and GRU) is used for annotation.
- As a condition for annotating the review:
    - All three classifiers must output same decision.
    - Each classifier must have accuracy > THRESHOLD.

- Each annotated review is added to training dataset of next iteration.
- To solve imbalance issues in training dataset Augmentation using Arabert **fill-mask** task is used before each iteration.
- To speedup convergence Threshold decay is used.
  
**Pros**: more accurate as labels is based on 3 opinions instead of 1.

**Cons**: Slower as it requires training 3 classifiers and using data augmentation with Arabert and requires high resources.


## Results & Discussion
- Starting with around ~6400 unlabeled data and running the loop for 5 iterations we found that:
    - about ~1000 review was still unlabeled.
    - only about ~400 review from total ~5400 were mislabeled.
- Using **LIME** Framework to explain the predictions for each model on both types of reviews we found:
    - Some of the mislabeled reviews maybe wrong labeled from the beginning -in the original dataset- like this example:

      ![Capture](https://github.com/MohamedMamdouh18/Semi-Supervised-Pseudo-Labeling-Arabic-Dataset/assets/63814228/39fa1caa-c17d-4228-b7e0-45426fcbb524)

    here we can see that the original label was 0 -negative- and the ensemble of the three models predicted 1 -positive-
  
    - Some of the left unlabeled reviews was vague and unclear and some of them seemed to be neutral so the models didn't agree on one prediction like this example:

      ![Capture](https://github.com/MohamedMamdouh18/Semi-Supervised-Pseudo-Labeling-Arabic-Dataset/assets/63814228/8c92b8c1-a634-4855-a3d6-d45df8acad6b)
      
      - GRU Model predicted 0 -negative-
      - RL & Linear SVM predicted 1 -positive-

## Authors

- [@Mohamed Mamdouh](https://github.com/MohamedMamdouh18)
- [@Mustafa El-Tawy](https://github.com/Mostafa424)
- [@Omar Khairat](https://github.com/OmarKhairat)
