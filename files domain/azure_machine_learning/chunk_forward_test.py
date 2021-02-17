'''
import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback
'''




import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import DistilBertTokenizer
from transformers import DistilBertModel, DistilBertPreTrainedModel, DistilBertForSequenceClassification, AdamW, DistilBertConfig
from transformers import get_linear_schedule_with_warmup

#enc = BertTokenizer.from_pretrained("bert-base-uncased")
enc = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

# Tokenizing input text

# this is question answering so change it only a single sentence
#text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
text = "[CLS] Who was Jim Henson ?"


tokenized_text = enc.tokenize(text)

# for deubg
print("tokenized_text: {}".format(tokenized_text))

# Masking one of the input tokens
# this is question answering so change it only a single sentence
#masked_index = 8
masked_index = 3

tokenized_text[masked_index] = '[MASK]'
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)

print("indexed_tokens: {}".format(indexed_tokens))
segments_ids = [0]

# Creating a dummy input
# but you need to move tensors to GPU
#https://github.com/huggingface/transformers/issues/227
# discuss convertion
#https://discuss.pytorch.org/t/best-way-to-convert-a-list-to-a-tensor/59949/2
#torch.tensor
#tokens_tensor = torch.tensor([indexed_tokens])
#segments_tensors = torch.tensor([segments_ids])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("create input for device: {}".format(device))
tokens_tensor = torch.tensor([indexed_tokens]).to(device)
segments_tensors = torch.tensor([segments_ids]).to(device)
dummy_input = tokens_tensor


# for debug
print("tokens_tensor shape for chunk: {}".format(tokens_tensor[0].shape[0])) 
for token_tensor in tokens_tensor:
    print("token_tensor shape for chunk: {}".format(token_tensor.shape[0]))



assert all(
        token_tensor.shape[0] == tokens_tensor[0].shape[0] for token_tensor in tokens_tensor
    ), "All input tenors have to be of the same shape"

# for deubg
# 14 tokens for output
print("tokens_tensor shape: {}".format(tokens_tensor.shape))
print("segments_tensor shape: {}".format(segments_tensors.shape))

print("tokens_tensor: {}".format(tokens_tensor))
print("segments_tensor: {}".format(segments_tensors))


# Initializing the model with the torchscript flag
# Flag set to True even though it is not necessary as this model does not have an LM Head.
#config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#    num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, torchscript=True)

# Instantiating the model
#model = BertModel(config)

# The model needs to be in evaluation mode
#model.eval()

# If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
#model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)
# classfication only  0 and 1 so set ut to 2
num_labels = 2

'''
class DistilBertForSequenceClassificationFilesDomain(DistilBertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassificationFilesDomain.from_pretrained('distilbert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config, weight=None):
        super(DistilBertForSequenceClassificationFilesDomain, self).__init__(config)
        self.num_labels = config.num_labels
        self.weight = weight

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None):
        # this line is related to wrong message
        distilbert_output = self.distilbert(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            head_mask=head_mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss(weight=self.weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

#Here we instantiate our model class. 
#We use a compact version, that is trained through model distillation from a base BERT model and modified to include a classification layer at the output. This compact version has 6 transformer layers instead of 12 as in the original BERT model.
# this class class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
# 
#https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/modeling_distilbert.html
# replace with my defined class but the same parameters
model = DistilBertForSequenceClassificationFilesDomain.from_pretrained('distilbert-base-uncased', num_labels=num_labels,
                                                            output_attentions=False, output_hidden_states=False)
'''

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels,
                                                            output_attentions=False, output_hidden_states=False)








#torch.onnx.export(model, dummy_input, 'traced_distill_bert.onnx', verbose=True)


#follow yue's suggestion to add output
# # ouput is slightly different, not sure it is related to 'do_constant_folding' for optimization for other parameter 

torch.onnx.export(model=model,
    args=(dummy_input),
    f='traced_distill_bert.onnx.bin',
    input_names = ["input_ids"],
    verbose=True,
    output_names = ["domain_output"],
    do_constant_folding = True,
    opset_version=11,
    dynamic_axes = {'input_ids': {1: '?'}, 'domain_output': {1: '?'}}
    )
