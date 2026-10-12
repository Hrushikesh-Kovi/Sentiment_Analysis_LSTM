# LSTM-Based Sentiment Analysis using PyTorch

## Overview

After implementing a 5-class sentiment analysis model using a vanilla Recurrent Neural Network (RNN), I explored **Long Short-Term Memory (LSTM)** networks to address the challenges associated with learning long-term dependencies in sequential text data.

Vanilla RNNs process sequences by maintaining a hidden state that carries information from previous time steps. However, they can struggle to preserve useful information over long sequences because of the vanishing gradient problem.

To explore an alternative architecture, I implemented an LSTM-based sentiment classifier using PyTorch and evaluated it on the same Corona sentiment analysis dataset used in my previous RNN project.

The LSTM achieved **75% test accuracy, approximately 10 percentage points higher than my previous RNN model**, demonstrating an improvement in performance on this task.

This project focuses on understanding LSTM architecture, implementing sequence classification with PyTorch, comparing recurrent architectures, and identifying overfitting through training and validation loss curves.

## Objectives

- Understand the limitations of vanilla RNNs when processing sequential text.
- Implement an LSTM network using PyTorch.
- Learn how the cell state and gating mechanisms regulate information flow.
- Convert text into token sequences and learn dense word representations.
- Perform 5-class sentiment classification.
- Compare LSTM performance with a previously implemented RNN baseline.
- Analyze training and validation loss to identify overfitting.
- Identify potential regularization techniques for future experiments.

## Dataset

**Dataset:** Corona Sentiment Analysis Dataset

**Dataset link:** [Insert Dataset Link]

The dataset contains tweet text and its corresponding sentiment label.

| Column | Description |
|---|---|
| `OriginalTweet` | Text input to the sentiment analysis model |
| `Sentiment` | Target sentiment category |

The task is to predict the sentiment associated with each tweet.

### Sentiment Classes

The dataset contains five sentiment categories:

1. Extremely Negative
2. Negative
3. Neutral
4. Positive
5. Extremely Positive

These categories are encoded as numerical class labels before being passed to the model.

The same dataset was used for the previous RNN implementation to maintain consistency in the comparison.

## Model Architecture

The model consists of three principal components:

1. Embedding layer
2. LSTM layer
3. Fully connected classification layer

### Architecture Configuration

| Parameter | Value |
|---|---|
| Embedding dimension | 100 |
| Hidden size | 128 |
| Number of LSTM layers | 1 |
| Direction | Unidirectional |
| Number of output classes | 5 |
| Framework | PyTorch |

### 1. Embedding Layer

The input text is tokenized and converted into sequences of integer token IDs using a vocabulary.

The embedding layer maps each token ID to a dense vector of 100 dimensions.

Instead of processing raw text or treating token IDs as meaningful numerical values, the model learns vector representations that can capture useful patterns during training.

For a batch of 32 sequences, each padded to 100 tokens, the input and embedding shapes would be:

- Token IDs: `(32, 100)`
- Embedded sequences: `(32, 100, 100)`

The actual batch size and sequence length depend on the data pipeline configuration.

### 2. LSTM Layer

The embedded sequences are passed to a single-layer, unidirectional LSTM with a hidden size of 128.

At every time step, the LSTM receives the current input representation and the previous hidden and cell states. It updates its internal states using three gates and a candidate cell state.

Unlike a vanilla RNN, an LSTM maintains a dedicated cell state to help preserve relevant information across time steps.

#### Cell State

The cell state carries information through the sequence and is updated using the forget gate, input gate, and candidate cell content.

#### Forget Gate

Controls how much of the previous cell state should be retained or discarded.

#### Input Gate

Controls how much new candidate information should be incorporated into the cell state.

#### Output Gate

Controls how much of the updated cell-state information is exposed through the hidden state.

These mechanisms help the network learn which information to preserve, update, and expose while processing a sequence.

LSTM gating can mitigate the vanishing gradient problem, although it does not eliminate it completely.

### 3. Fully Connected Layer

The final hidden representation is passed to a linear layer that produces five output logits, corresponding to the five sentiment classes.

The predicted class is obtained by selecting the class with the highest logit.

When using `nn.CrossEntropyLoss`, the model returns raw logits without applying softmax in the forward pass.

## How the Model Works

The overall workflow is:

1. Load and inspect the dataset.
2. Clean and preprocess the tweet text.
3. Tokenize the text and construct a vocabulary.
4. Convert tokens into numerical sequences.
5. Pad or truncate sequences to the selected length.
6. Create training, validation, and test datasets.
7. Load batches using PyTorch `DataLoader`.
8. Convert token IDs into dense embeddings.
9. Process embedded sequences using the LSTM.
10. Pass the final hidden representation through the classification layer.
11. Calculate classification loss and update model parameters during training.
12. Evaluate the model and analyze its performance.

## Results

The model achieved the following test accuracy:

| Metric | Result |
|---|---:|
| Test accuracy | 75% |
| Improvement over previous RNN | Approximately 10 percentage points |

The LSTM outperformed my previous vanilla RNN implementation on the same sentiment classification task.

This comparison provides an initial indication that the LSTM architecture was better suited to the task under the configurations tested.

However, accuracy alone does not provide a complete picture of model performance. Per-class precision, recall, F1-score, and a confusion matrix can provide additional insight into which sentiment categories are being predicted correctly.

The comparison should also be interpreted in the context of the models' respective hyperparameters, training procedures, and selected checkpoints.

## Training and Validation Loss Analysis

Training and validation loss curves were plotted to monitor the learning process and identify potential overfitting.

### Observations

- Training loss decreased consistently throughout training.
- Validation loss initially decreased, reaching its minimum around the third epoch.
- After that point, validation loss increased while training loss continued to decrease.

This divergence suggests that the model was increasingly fitting the training data without improving its generalization to the validation data.

### Early Stopping

Early stopping was used to monitor validation performance and retain the checkpoint corresponding to the best validation loss.

This prevents the final model from automatically being selected merely because it was trained for the largest number of epochs.

However, early stopping does not eliminate overfitting itself; it helps select a checkpoint that may generalize better.

## Challenges and Future Improvements

The primary challenge observed during training was overfitting.

The following experiments are planned to investigate ways to improve generalization.

### 1. Dropout Regularization

Dropout can reduce reliance on specific activations by randomly disabling a proportion of them during training.

A dropout layer can be applied to the LSTM's final hidden representation before the classification layer.

### 2. AdamW and Weight Decay

AdamW with weight decay can help regularize the model by discouraging excessively large weights.

The learning rate and weight-decay coefficient will need to be tuned using validation performance.

### 3. Data Augmentation

Meaning-preserving text augmentation may increase the diversity of training examples.

Any augmented examples must preserve their sentiment labels, and augmentation should be applied only to the training set.

### 4. Reduce Model Complexity

Reducing the LSTM hidden size or simplifying the architecture may help limit the model's capacity to overfit.

The effect of smaller architectures should be measured rather than assumed.

### 5. Further Evaluation

Additional evaluation will include:

- Classification report with per-class precision, recall, and F1-score.
- Confusion matrix analysis.
- Comparison of training and validation curves across experiments.
- Consistent evaluation on a held-out test set.
- Experiments to assess the impact of sequence lengths and padding.

These improvements are proposed next steps and have not been presented as completed experiments.

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn

## Key Learnings

Through this project, I explored:

- The limitations of vanilla RNNs when learning long-term dependencies.
- The role of cell states and gating mechanisms in LSTMs.
- Implementing recurrent sequence classification in PyTorch.
- Comparing recurrent architectures on the same NLP task.
- Interpreting training and validation loss curves.
- Using early stopping to select a model checkpoint.
- Identifying overfitting and planning controlled regularization experiments.

## Conclusion

This project was a step forward in my journey into deep learning and natural language processing.

By implementing an LSTM-based sentiment classifier and comparing it with my earlier vanilla RNN model, I observed improved test accuracy and gained practical experience with recurrent architectures, sequence modeling, and model evaluation.

The next phase will focus on reducing overfitting, evaluating additional metrics, and measuring the effect of regularization techniques through controlled experiments.

**The goal is not just to build a model that performs well on training data, but to build one that generalizes reliably to unseen text.**

## Connect

LinkedIn: https://www.linkedin.com/in/kovihrushikesh/
