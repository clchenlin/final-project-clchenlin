import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset, Features, ClassLabel
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

#****************** Functions for data preparation ******************#

# preprocess the dataset
def preprocess(dataset):
    '''
    Preprocess the dataset

    Args:
        dataset (pandas.DataFrame): dataset to preprocess
    
    Returns:
        dataset (pandas.DataFrame): dataset with standardized column names
    '''
    # drop article link and rename column
    dataset = dataset.drop(['article_link'],axis=1)
    dataset = dataset.rename(columns={'headline':'text', 'is_sarcastic':'label'})

    # shuffle data
    dataset = dataset.sample(frac=1, random_state=123).reset_index(drop=True)

    return dataset

# create plot to check for class imbalance
def plot_class(dataset):
    '''
    Create plot to check for class imbalance
    
    Args:
        dataset (pandas.DataFrame): headline dataset
        
    Returns:
        None
    '''
    plt.figure(figsize=(4,3))
    ax = sns.countplot(x="label", data=dataset)
    ax.set_xticklabels(['nonsarcastic', 'sarcastic'])
    plt.xlabel('Classification')
    plt.show()



#****************** Functions for feature extraction ******************#

# tokenize batch
def tokenize(batch, tokenizer):
    return tokenizer(batch["text"],
                     truncation=True,
                     max_length=512,
                     padding=True)


# encode text using AutoTokenizer
def encode(dataset, tokenizer):
    '''
    Encode text using AutoTokenizer

    Args:
        dataset (DatasetDict): dataset to encode
        tokenizer (AutoTokenizer): tokenizer from pretrained model to use
    Returns:
        dataset_encoded (torch.Tensor): encoded dataset
    '''
    dataset_encoded = dataset.map(tokenize, 
                                  batched=True,
                                  batch_size=None,
                                  fn_kwargs={"tokenizer": tokenizer})
    dataset_encoded.set_format('torch')

    return dataset_encoded


# extract last hidden states from encoded dataset
def extract_states(batch, model, postprocess, device):
    '''
    Extract last hidden states from encoded dataset

    Args:
        batch (dict): batch of encoded dataset
        model (AutoModel): pretrained model to use
        postprocess (str): postprocessing method to use. 'average', 'last', or 'first'.
    Returns:
        batch (dict): batch of encoded dataset with last hidden states
    '''
    input_ids = torch.tensor(batch["input_ids"]).to(device)
    attention_mask = torch.tensor(batch["attention_mask"]).to(device)

    if postprocess=='average':
        with torch.no_grad():
            output = model(input_ids, attention_mask)
            lhs = output.last_hidden_state.cpu().numpy()

        valid = np.array(batch["attention_mask"]).astype(bool)

        batch_size, n_tokens, hidden_dim = lhs.shape
        valid = valid.reshape(batch_size, n_tokens, 1).repeat(hidden_dim, axis=-1)
        # average over the valid tokens
        masked_mean = np.ma.array(lhs, mask=~valid).mean(axis=1).data

        batch["hidden_state"] = masked_mean

    elif postprocess=='last':
        with torch.no_grad():
                output = model(input_ids, attention_mask)
                lhs = output.last_hidden_state.cpu().numpy()

        valid = np.array(batch["attention_mask"]).astype(bool)

        batch_size, n_tokens, hidden_dim = lhs.shape
        valid = valid.reshape(batch_size, n_tokens, 1).repeat(hidden_dim, axis=-1)

        lhs_m = np.ma.array(lhs, mask=~valid)

        # find the indices of unmasked values along axis 1 
        valid_indices = np.ma.notmasked_edges(lhs_m, axis=1)

        # select the last hidden states of the last valid token
        last_valid = lhs_m[valid_indices[-1]].data.reshape(batch_size,-1)

        batch["hidden_state"] = last_valid
    
    else:
        with torch.no_grad():
            output = model(input_ids, attention_mask)
            # select the last hidden states of the first token
            lhs = output.last_hidden_state[:,0].cpu().numpy()

        batch["hidden_state"] = np.array(lhs)
    
    return batch


#****************** Functions for fine-tuning ******************#

# compute metrics
def compute_metrics(pred):
    logits, labels = pred
    preds = logits.argmax(axis=-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}



#****************** Functions for fine-tuning with peft ******************#

# print trainable parametersdef print_trainable_parameters(model):
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.

    Args:
        model: model to be fine-tuned
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# apply the fine-tuned model to the test dataset
def apply_to_test(trainer, test_dataset):
    """
    Apply the fine-tuned model to the test dataset.

    Args:
        trainer: trainer object
        test_dataset: test dataset
    Returns:
        y_preds: predictions on the test dataset
    """
    preds_output = trainer.predict(test_dataset)
    y_preds = np.argmax(preds_output.predictions, axis=1)

    return y_preds

# plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels, title):
    '''
    Plot confusion matrix
    
    Args:
        y_true (list): true labels
        y_pred (list): predicted labels
        labels (list): list of labels
        title (str): title of plot
    
    Returns:
        None
    '''
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)
    disp.plot()
    plt.title(title)
    plt.show()