# Text Classification Enhancement Using PEFT on DistilBERT

## Overview
This repository contains the work and findings of a project aimed at improving text classification using the `distilbert-base-uncased` model on the 'ag_news' dataset through Parameter-Efficient Fine-Tuning (PEFT).

## Objective
To demonstrate how PEFT can significantly boost the performance of a DistilBERT model in text classification tasks, while dramatically reducing the number of trainable parameters.

## Dataset
The project utilizes the 'ag_news' dataset, comprising various news articles for text classification.


## Methodology
- Utilization of `distilbert-base-uncased` as the base model.
- Application of PEFT techniques using `PeftModel`, `PeftConfig`, `get_peft_model`, and `LoraConfig`.
- Focus on reducing the trainable parameters while enhancing model accuracy.

## Results
- Initial F1-Score: 0.206
- Post-PEFT F1-Score: 0.9194
- Reduction in trainable parameters leads to more efficient training without compromising accuracy.
- The application of PEFT has led to a remarkable improvement in the model's performance. The F1-score, a measure of a test's accuracy, increased from a modest 0.206 to an impressive 0.9194. This substantial increase highlights the efficiency of fine-tuning a select few parameters in contrast to adjusting the entire set of parameters in the BERT model.

## Training Parameters:
- Total Parameters in DistilBERT: 67,587,080
- Trainable Parameters with PEFT: 630,532
- Percentage of Trainable Parameters: ~0.93%

## Conclusion
This project illustrates the potential of PEFT in enhancing the capabilities of large language models like DistilBERT, particularly in the context of text classification. By reducing the number of trainable parameters without compromising on performance, we open doors to more efficient and accessible NLP model training, especially for scenarios with limited computational resources.

## How to Use
Detailed instructions on setting up the environment, loading the dataset, and running the model are provided in the respective folders and scripts.

## Contributions
Contributions are welcome. Please read the contribution guidelines for more information.
