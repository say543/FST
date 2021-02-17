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

from transformers import DistilBertPreTrainedModel, DistilBertModel
from transformers import DistilBertTokenizer,DistilBertTokenizerFast
from transformers import DistilBertForTokenClassification, AdamW, DistilBertConfig

from transformers import get_linear_schedule_with_warmup
from transformers import BatchEncoding
from tokenizers import Encoding




#########################
# torch tensor operation test below
#########################
'''
output = torch.tensor([[[-0.1221, -0.3479, -0.0684,  0.0110,  0.2062,  0.1621, -0.0185,
           0.0124,  0.5029,  0.7174,  0.2862, -0.2431,  0.1328, -0.1321,
          -0.4819,  0.2302,  0.0615, -0.2918, -0.3064, -0.2910,  0.3069],
         [ 0.0583, -0.2261, -0.1034,  0.1108,  0.0693,  0.2041, -0.1494,
          -0.0275,  0.0287,  0.0484,  0.3371, -0.0232,  0.2029,  0.0085,
          -0.2478,  0.1623,  0.0651, -0.0443, -0.4363, -0.0838,  0.0469],
         [ 0.0049, -0.2861, -0.0598,  0.1419, -0.0236,  0.2379, -0.1346,
          -0.1205,  0.0782,  0.0839,  0.2446,  0.0358,  0.2748, -0.0316,
          -0.1360,  0.1410,  0.0341, -0.1465, -0.4751, -0.1850,  0.0027],
         [-0.2048, -0.2630,  0.1079, -0.1364, -0.0478,  0.2168, -0.2872,
          -0.0433,  0.2038, -0.0138,  0.2251, -0.1251, -0.0528,  0.0448,
          -0.3945,  0.2250,  0.0129,  0.0118, -0.3394,  0.0837,  0.0479],
         [ 0.3469,  0.1035,  0.2902,  0.1426,  0.0818, -0.1072,  0.3632,
           0.0375,  0.3482, -0.2737,  0.0239, -0.1121, -0.1262, -0.2249,
          -0.2885, -0.0055, -0.3414,  0.1324, -0.1100, -0.1086,  0.0097]]])

for i, ele2d in enumerate(output):
    for j, ele1d in enumerate(ele2d):
        value, index = torch.max(ele1d, dim=0)
        print("value: {} anmd index {}".format(value, index))
'''

#########################
# torch tensor operation test above
#########################


#enc = BertTokenizer.from_pretrained("bert-base-uncased")
fast_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased') # Load a pre-trained tokenizer

# Tokenizing input text

# save onnx
# Tokenizing input text
text = "a visually stunning rumination on love"
fast_tokenized_batch : BatchEncoding = fast_tokenizer(text)
fast_tokenized_text :Encoding  =fast_tokenized_batch[0]
fast_tokenized_text_tokens_copy = fast_tokenized_text.tokens

print("fast token ouput version 2 being used here: {}".format(fast_tokenized_text_tokens_copy))

# Masking one of the input tokens
# this is question answering so change it only a single sentence
#masked_index = 8
masked_index = 3
fast_tokenized_text_tokens_copy[masked_index] = '[MASK]'
indexed_tokens = fast_tokenizer.convert_tokens_to_ids(fast_tokenized_text_tokens_copy)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for debug 
print("device: {}".format('cuda' if torch.cuda.is_available() else 'cpu'))

print("indexed_tokens: {}".format(indexed_tokens))
segments_ids = [0]

# Creating a dummy input
# but you need to move tensors to GPU
#https://github.com/huggingface/transformers/issues/227
#tokens_tensor = torch.tensor([indexed_tokens])
#segments_tensors = torch.tensor([segments_ids])
print("create input for device: {}".format(device))

# adding [] is the same as unqueeze(0) function
tokens_tensor = torch.tensor([indexed_tokens]).to(device)

segments_tensors = torch.tensor([segments_ids]).to(device)
dummy_input = tokens_tensor

# for deubg
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



# read label
from typing_extensions import TypedDict
from typing import List,Any
IntList = List[int] # A list of token_ids
IntListList = List[IntList] # A List of List of token_ids, e.g. a Batch


import itertools
class LabelSet:
    def __init__(self, labels: List[str]):
        self.labels_to_id = {}
        self.ids_to_label = {}

        self.labels_to_id["o"] = 0
        self.ids_to_label[0] = "o"
        num = 1
        for label in labels:
            if label == "o":
                print("skip:{}".format(label))
                continue
            self.labels_to_id[label] = num
            self.ids_to_label[num] = label
            num = num +1 


    def get_aligned_label_ids_from_aligned_label(self, aligned_labels):
        return list(map(self.labels_to_id.get, aligned_labels))

    def get_untagged_id(self):
        return self.labels_to_id["o"]

    def get_labels(self):
        return self.labels_to_id

slots = ["O", 
    "file_name", 
    "file_type", 
    "data_source", 
    "contact_name", 
    "to_contact_name",
    "file_keyword",
    "date",
    "time",
    "meeting_starttime",
    "file_action",
    "file_action_context",
    "position_ref",
    "order_ref",
    "file_recency",
    "sharetarget_type",
    "sharetarget_name",
    "file_folder",
    "data_source_name",
    "data_source_type",
    "attachment"]

# map all slots to lower case
slots_label_set = LabelSet(labels=map(str.lower,slots))

num_labels = len(set(slots_label_set.get_labels()))






#model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels,
#                                                            output_attentions=False, output_hidden_states=False)


class DistilBertForTokenClassificationFilesDomain(DistilBertPreTrainedModel):
    """
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
        super(DistilBertForTokenClassificationFilesDomain, self).__init__(config)
        self.num_labels = config.num_labels
        self.weight = weight

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None):
        # this line is related to wrong message
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output


#Here we instantiate our model class. 
#We use a compact version, that is trained through model distillation from a base BERT model and modified to include a classification layer at the output. This compact version has 6 transformer layers instead of 12 as in the original BERT model.
# this class class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
# 
#https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/modeling_distilbert.html
# replace with my defined class but the same parameters
model = DistilBertForTokenClassificationFilesDomain.from_pretrained('distilbert-base-uncased', num_labels=num_labels,
                                                            output_attentions=False, output_hidden_states=False)





##################################################
# Store model(not in pytorch) and load it back - below
##################################################


output_dir = './slot_model_saved_test/'
import os, argparse
# if folder does not exist then create
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# use `save_pretrained()`  to preserve model (one function),，config(same as preserve function), 
# tokenizer
# (verisin 1 : two functions, tokenizer.save_vocabulary() / model_to_save.config.to_json_file)
# (version 2 : a function for files :  vocab.txt , tokenizer_config.json)
# they can be loaded `from_pretrained()
# considering distributed/parallel training 
# If we have a distributed model, save only the encapsulated model
# ? not sure how it realy means
model_to_save = model.module if hasattr(model, 'module') else model  
model_to_save.save_pretrained(output_dir)
fast_tokenizer.save_pretrained(output_dir)

# save your training arguments together with the trained model
# originlal those are from input argument
# here setup value to mimic
# learning_rate = args.learning_rate
# adam_epsilon = args.adam_epsilon
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, dest='dataset_name', default='')
parser.add_argument('--batch_size', type=int, dest='batch_size', default=32)
parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=1e-5)
parser.add_argument('--adam_epsilon', type=float, dest='adam_epsilon', default=1e-8)
parser.add_argument('--num_epochs', type=int, dest='num_epochs', default=5)
args = parser.parse_args()
args.learning_rate = 2e-5
args.adam_epsilon = 1e-8
torch.save(args, os.path.join(output_dir, 'training_args.bin'))


# load model back
# you need to know exact class for each one
# it expects file like pytorch_model.bin
# <failed >version 1: so rename *pt to see if it wokrs. it will complaint 
#torch.nn.modules.module.ModuleAttributeError: 'DistilBertForTokenClassification' object has no attribute 'keys'
# version2 : load pytroch_model.bin (not sure if it is after trained or not)
model = DistilBertForTokenClassification.from_pretrained(output_dir)
# load tokenizer back
# you need to know exact class for each one
fast_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased') # Load a pre-trained tokenizer

# copy model to  GPU/CPU to work
model.to(device)


print("load model done: !")


print("evaluate model ....")

test_input_tensor = torch.tensor([[101, 12453, 19453, 6254, 102]])
at_mask_tensor = torch.tensor([[1, 1, 1, 1, 1]])
label_mask_tensor = torch.tensor([[0, 0, 6, 0, 0]])
with torch.no_grad():
    model.eval()

    outputs = model(test_input_tensor, attention_mask=at_mask_tensor, labels=label_mask_tensor)
            
    loss = outputs[0]


##################################################
# Store model(not in pytorch) and load it back - above
##################################################


'''
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
'''