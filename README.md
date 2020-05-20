# NLP MCQA Project

## Introduction

We are required to develop a multiple-choice document-based question answering system to select the answer from several candidates. 

Input: a document, and a question (query) 

Output: an answer (select from options)

We are provided three multiple-choice document-based question answering dataset to
evaluate our QA system, i.e., MCTest, RACE, and DREAM. 

To implement our QA system, we managed to learn the approach from [Jin, Di, Shuyang Gao, Jiun-Yu Kao, Tagyoung Chung, and Dilek Hakkani-tur. "MMM: Multi-stage Multi-task Learning for Multi-choice Reading Comprehension." AAAI (2020).](https://arxiv.org/pdf/1910.00458.pdf). We are applying Multi-stage Multi-task Learning method on Bert Large Model to train and test our model.


```
@article{jin2019mmm,
  title={MMM: Multi-stage Multi-task Learning for Multi-choice Reading Comprehension},
  author={Jin, Di and Gao, Shuyang and Kao, Jiun-Yu and Chung, Tagyoung and Hakkani-tur, Dilek},
  journal={arXiv preprint arXiv:1910.00458},
  year={2019}
}
```

## Requirements
### Python packages
- Pytorch
- Python 3.69 

## Usage
1. All five MCQA datasets are put in the folder "data" and to unzip the RACE data, run the following command:
```
tar -xf RACE.tar.gz
```

2. To train the BERT model (including base and large versions), use the following command:

```
python run_classifier_bert_exe.py TASK_NAME MODEL_DIR BATCH_SIZE_PER_GPU GRADIENT_ACCUMULATION_STEPS
```
Here we explain each required argument in details:
- TASK_NAME: It can be a single task or multiple tasks. If a single task, the options are: dream, race, toefl, mcscript, mctest160, mctest500, mnli, snli, etc. Multiple tasks can be any combinations of those above-mentioned single tasks. For example, if you want to train a multi-task model on the dream and race tasks together, then this variable should be set as "dream,race".
- MODEL_DIR: Model would be initialized by the parameters stored in this directory. In this project, we focus on Bert-Large-Uncased model.
- BATCH_SIZE_PER_GPU: Batch size of data in a single GPU.
- GRADIENT_ACCUMULATION_STEPS: How many steps to accumulate the gradients for one step of back-propagation.

One note: the effective batch size for training is important, which is the product of three variables: BATCH_SIZE_PER_GPU, NUM_OF_GPUs, and GRADIENT_ACCUMULATION_STEPS. It is recommended to be at least higher than 12 and 24 would be great. 

For BERT-Large, 16 GB GPU (which is the maximum memory size for Cloud GPU in Colab) cannot hold a single batch since each data sample is composed of four choices which comprise of 4 sequences of 512 max_sequence_length. So in order to put a single batch to a 16 GB GPU, we need to decrease the max_sequence_length from 512 to some number smaller, although this will degrade the performance by a little.

3. To test the BERT model (including base and large versions), use the following command:

```
python test_classifier_bert_exe.py TASK_NAME MODEL_DIR 
```
If we have already trained and saved the model, we can simply load the model state to a pretrained model so that we can have a fine-tuned model for prediction.

The saved model state ("pytorch_model.bin") have to be placed at "/content/MCQA_project/tmp/{task}_{model_dir}" folder, e.g. "race_bert-large-uncased"

4. To predict the answer by the BERT model (including base and large versions), use the following command:

```
python pred_classifier_bert_exe.py TASK_NAME MODEL_DIR 
```
Prediction Demostration - MCTest (mctest160 and mctest500)
Step 1: Run the command and system will randomly draw a sample from test.tsv

Step 2: The prediction results and answers will be shown.

Step 3: Change the random seed in pred_classifier_bert_exe.py to select another sample

Prediction Demostration - Race
Step 1: Create a "demo" folder in "test" folder, which is the input folder

Step 2: Randomly select a txt file from "test" folder, which includes both the document and question

Step 3: Put the selected file into the "demo" folder

Step 4: Run the command and it will show the prediction results and answers

Prediction Demostration - Dream
Step 1: Run the command and system will randomly draw a sample from test.json

Step 2: The prediction results and answers will be shown.

Step 3: Change the random seed in pred_classifier_bert_exe.py to select another sample



