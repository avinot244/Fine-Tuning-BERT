import torch
from transformers import BertTokenizer, BertModel

import logging
import matplotlib.pyplot as plt

tokenizer : BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "Aatrox is a good champion for the toplane role and is also one of the top champions in the tierlists"

# Add the special markers
marked_text = "[CLS]" + text + " [SEP]"

# Split the sentence into tokens.
tokenized_text = tokenizer.tokenize(marked_text)

# Map the token strings to their vocabulary indeces
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Mark each of the 22 tokens as belonging to sentence "1".
segments_ids = [1] * len(tokenized_text)

tokens_tensor : torch.Tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])


# Loading our bert model
model : BertModel = BertModel.from_pretrained("bert-base-uncased",
                                  output_hidden_states = True)

model.eval()

# Run the text through BERT, and collect all of the hidden states produced
# from all 12 layers.
with torch.no_grad():
    
    outputs = model(tokens_tensor, segments_tensors)

    # Evaluating the model will return a different number of objects based on 
    # how it's  configured in the `from_pretrained` call earlier. In this case, 
    # becase we set `output_hidden_states = True`, the third item will be the 
    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel

    hidden_states = outputs[2]

# hidden_states is a 4 dimensional array that stores
# The Layer Number
# The batch number (1 in our case because we have one sentance)
# The word/token number
# The hidden unit/feature number

print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
layer_i = 0

print ("Number of batches:", len(hidden_states[layer_i]))
batch_i = 0

print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

toekn_i = 5
layer_i = 5
vec = hidden_states[layer_i][batch_i][token_i]

plt.figure(figsize=(10, 10))
plt.hist(vec, bins=200)
plt.savefig("Range_of_values_of_the_hidden_unit.png")
