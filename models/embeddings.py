"""
doc string for the InputEmbedding class and its methods
To do
"""
import numpy as np


class InputEmbedding:
    """
    Description of the InputEmbedding class and its methods
    To do
    """
    def __init__(self):
        print("InputEmbedding initialized")
    def input_embedding(self,inputs_sequences,embedding_dict):
        """
        Convert the input batch of strings to its corresponding embedding vector
        
        parameters:
        inputs: list of strings, the input batch, the length of the strings is seq_len
        Returns:
        numpy array of shape (batch_size,seq_len,d_model)
        """
        batch_sequences = []
        for input_seq in inputs_sequences:
            single_sequence = []
            for word in input_seq:
                if word not in embedding_dict:
                    raise ValueError(f"{word} not found in the embedding dictionary")
                single_sequence.append(embedding_dict[word])
            batch_sequences.append(single_sequence)
        # return a tensor of shape (batch_size,seq_len,d_model)
        return np.array(batch_sequences)
