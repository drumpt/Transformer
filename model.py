import os
import math

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Transformer(nn.Module):
    def __init__(self, model_dim, src_vocab_size, tgt_vocab_size, max_length):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_length = max_length
        self.model_dim = model_dim
        # self.num_identical_layers = 6
        self.h = 8
        self.positional_encoding_constants = self.get_positional_encoding_constants()

        self.input_embedding = nn.Embedding(self.src_vocab_size, self.model_dim)
        self.output_embedding = nn.Embedding(self.tgt_vocab_size, self.model_dim)
        self.embedding_dropout = nn.Dropout(p = 0.1)
        self.encoder = Encoder(self.model_dim, self.h)
        self.decoder = Decoder(self.model_dim, self.h)
        self.final_linear = nn.Linear(self.model_dim, self.tgt_vocab_size)
        self.softmax = nn.Softmax(dim = 2)

    def forward(self, x, y): # x : encoder_input, y : decoder_input
        encoder_input = self.embedding_dropout(self.positional_encoding(self.input_embedding(x)))
        encoder_output = self.encoder(encoder_input)
        decoder_output = self.decoder(self.embedding_dropout(self.positional_encoding(self.output_embedding(y))), encoder_output)
        final_output = self.softmax(self.final_linear(decoder_output))
        return final_output

    def positional_encoding(self, embedded_sentence): # TODO(completed): batch 단위가 아니라 문장 단위로 전달되는지 확인. 안 됨.
        for batch_idx in range(embedded_sentence.shape[0]):
            embedded_sentence[batch_idx] += self.positional_encoding_constants[:embedded_sentence.shape[1], :]
        return embedded_sentence

    def get_positional_encoding_constants(self): # avoid redundant calculations
        positional_encoding_constants = []
        for pos in range(self.max_length):
            pos_constants = []
            for idx in range(self.model_dim):
                if idx % 2 == 0: # idx = 2 * i -> 2 * i = idx
                    pos_constants.append(math.sin(pos / 10000 ** (idx / self.model_dim)))
                else: # idx = 2 * i + 1 -> 2 * i = idx - 1
                    pos_constants.append(math.cos(pos / 10000 ** ((idx  - 1) / self.model_dim)))
            positional_encoding_constants.append(pos_constants)
        return torch.tensor(positional_encoding_constants)
    
    def save_model(self, output_path, epoch, loss, val_loss):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_filename = os.path.join(output_path, f"weights_{epoch:03d}_{loss:.4f}_{val_loss:.4f}.pt")
        torch.save(self.state_dict(), output_filename)
        return output_filename

    def plot(self, output_path, history):
        plt.subplot(2, 1, 1)
        plt.title('Accuracy versus Epoch')
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.legend(['accuracy', 'val_accuracy'], loc = 'upper right')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.subplot(2, 1, 2)
        plt.title('Loss versus Epoch')
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.legend(['loss', 'val_loss'], loc = 'upper right')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "training_result.png"))

class Encoder(nn.Module):
    def __init__(self, model_dim, h):
        super().__init__()
        self.model_dim = model_dim
        self.h = h

        # TODO : num_identical_layers를 이용할 수 있을까?
        self.identical_layer1 = EncoderIdenticalLayer(self.model_dim, self.h)
        self.identical_layer2 = EncoderIdenticalLayer(self.model_dim, self.h)
        self.identical_layer3 = EncoderIdenticalLayer(self.model_dim, self.h)
        self.identical_layer4 = EncoderIdenticalLayer(self.model_dim, self.h)
        self.identical_layer5 = EncoderIdenticalLayer(self.model_dim, self.h)
        self.identical_layer6 = EncoderIdenticalLayer(self.model_dim, self.h)

    def forward(self, x):
        x = self.identical_layer1(x)
        x = self.identical_layer2(x)
        x = self.identical_layer3(x)
        x = self.identical_layer4(x)
        x = self.identical_layer5(x)
        x = self.identical_layer6(x)
        return x

class EncoderIdenticalLayer(nn.Module):
    def __init__(self, model_dim, h):
        super().__init__()
        self.model_dim = model_dim
        self.h = h

        self.w_i_Q_list = [nn.Linear(in_features = self.model_dim, out_features = self.model_dim // self.h, bias = False) for _ in range(self.h)]
        self.w_i_K_list = [nn.Linear(in_features = self.model_dim, out_features = self.model_dim // self.h, bias = False) for _ in range(self.h)]
        self.w_i_V_list = [nn.Linear(in_features = self.model_dim, out_features = self.model_dim // self.h, bias = False) for _ in range(self.h)]
        self.w_O = nn.Linear(in_features = self.h * (self.model_dim // self.h), out_features = self.model_dim, bias = False)
        self.layer_normalization1 = nn.LayerNorm(self.model_dim) # exclude batch dimension

        self.feed_forward_network1 = nn.Linear(in_features = 512, out_features = 2048, bias = True)
        self.feed_forward_network2 = nn.Linear(in_features = 2048, out_features = 512, bias = True)
        self.layer_normalization2 = nn.LayerNorm(self.model_dim) # TODO(completed): 맞나?

        self.dropout = nn.Dropout(p = 0.1)
        self.softmax = nn.Softmax(dim = 2) # TODO(completed): 이게 맞나? 2인 것 같다.
        self.relu = nn.ReLU()

    def forward(self, x):
        ### Sublayer 1
        # Multi-Head Attention
        splitted_x_list = []
        for i in range(self.h):
            in_softmax = torch.matmul(
                self.w_i_Q_list[i](x),
                self.w_i_K_list[i](x).transpose(1, 2)
            ) / math.sqrt(self.model_dim // self.h)
            out_softmax = self.w_i_V_list[i](x)
            splitted_x = torch.matmul(self.softmax(in_softmax), out_softmax)
            splitted_x_list.append(splitted_x)
        concatenated_x = torch.cat(splitted_x_list, dim = 2) # TODO(completed): 맞나?
        multi_head_attention_output = self.dropout(self.w_O(concatenated_x))

        # Add & Layer Normalization
        multi_head_attention_output += x
        multi_head_attention_output = self.layer_normalization1(multi_head_attention_output)

        ### Sublayer 2
        # Feed-Forward Network
        positionwise_output_list = [] # (0, 2, 1)
        for position in range(multi_head_attention_output.shape[1]): # token length
            positionwise_input = multi_head_attention_output[:, position, :]
            positionwise_output = self.relu(self.feed_forward_network1(positionwise_input))
            positionwise_output = self.feed_forward_network2(positionwise_output)
            positionwise_output_list.append(positionwise_output)
        ffn_output = self.dropout(torch.stack(positionwise_output_list, dim = 1))

        # Add & Layer Normalization
        ffn_output += multi_head_attention_output
        ffn_output = self.layer_normalization2(ffn_output)
        return ffn_output

class Decoder(nn.Module):
    def __init__(self, model_dim, h):
        super().__init__()
        self.model_dim = model_dim
        self.h = h

        self.identical_layer1 = DecoderIdenticalLayer(self.model_dim, self.h)
        self.identical_layer2 = DecoderIdenticalLayer(self.model_dim, self.h)
        self.identical_layer3 = DecoderIdenticalLayer(self.model_dim, self.h)
        self.identical_layer4 = DecoderIdenticalLayer(self.model_dim, self.h)
        self.identical_layer5 = DecoderIdenticalLayer(self.model_dim, self.h)
        self.identical_layer6 = DecoderIdenticalLayer(self.model_dim, self.h)

    def forward(self, x, y):
        x = self.identical_layer1(x, y)
        x = self.identical_layer2(x, y)
        x = self.identical_layer3(x, y)
        x = self.identical_layer4(x, y)
        x = self.identical_layer5(x, y)
        x = self.identical_layer6(x, y)
        return x

class DecoderIdenticalLayer(nn.Module):
    def __init__(self, model_dim, h):
        super().__init__()
        self.model_dim = model_dim
        self.h = h

        self.w_i_Q_list_first = [nn.Linear(in_features = self.model_dim, out_features = self.model_dim // self.h, bias = False) for _ in range(self.h)]
        self.w_i_K_list_first = [nn.Linear(in_features = self.model_dim, out_features = self.model_dim // self.h, bias = False) for _ in range(self.h)]
        self.w_i_V_list_first = [nn.Linear(in_features = self.model_dim, out_features = self.model_dim // self.h, bias = False) for _ in range(self.h)]
        self.w_O_first = nn.Linear(in_features = self.h * (self.model_dim // self.h), out_features = self.model_dim, bias = False)
        self.layer_normalization1 = nn.LayerNorm(self.model_dim) # exclude batch dimension

        self.w_i_Q_list_second = [nn.Linear(in_features = self.model_dim, out_features = self.model_dim // self.h, bias = False) for _ in range(self.h)]
        self.w_i_K_list_second = [nn.Linear(in_features = self.model_dim, out_features = self.model_dim // self.h, bias = False) for _ in range(self.h)]
        self.w_i_V_list_second = [nn.Linear(in_features = self.model_dim, out_features = self.model_dim // self.h, bias = False) for _ in range(self.h)]
        self.w_O_second = nn.Linear(in_features = self.h * (self.model_dim // self.h), out_features = self.model_dim, bias = False)
        self.layer_normalization2 = nn.LayerNorm(self.model_dim) # exclude batch dimension
        
        self.feed_forward_network1 = nn.Linear(in_features = 512, out_features = 2048, bias = True)
        self.feed_forward_network2 = nn.Linear(in_features = 2048, out_features = 512, bias = True)
        self.layer_normalization3 = nn.LayerNorm(self.model_dim)

        self.dropout = nn.Dropout(p = 0.1)
        self.softmax = nn.Softmax(dim = 2)
        self.relu = nn.ReLU()

    def forward(self, x, y): # TODO(completed): masking 구현.
        ### Sublayer 1
        # Masked Multi-Head Attention
        splitted_x_list = []
        for i in range(self.h):
            in_softmax = torch.matmul(
                self.w_i_Q_list_first[i](x),
                self.w_i_K_list_first[i](x).transpose(1, 2)
            ) / math.sqrt(self.model_dim // self.h)
            in_softmax = self.masking(in_softmax)
            out_softmax = self.w_i_V_list_first[i](x)
            splitted_x = torch.matmul(self.softmax(in_softmax), out_softmax)
            splitted_x_list.append(splitted_x)
        concatenated_x = torch.cat(splitted_x_list, dim = 2)
        multi_head_attention_output_first = self.dropout(self.w_O_first(concatenated_x))

        # Add & Layer Normalization
        multi_head_attention_output_first += x
        multi_head_attention_output_first = self.layer_normalization1(multi_head_attention_output_first)

        ## Sublayer 2
        # Multi-Head Attention
        """
        queries : come from previous decoder layer
        keys, values : come from the output of the encoder
        """
        splitted_x_list = []
        for i in range(self.h):
            in_softmax = torch.matmul(
                self.w_i_Q_list_second[i](multi_head_attention_output_first),
                self.w_i_K_list_second[i](y).transpose(1, 2)
            ) / math.sqrt(self.model_dim // self.h)
            out_softmax = self.w_i_V_list_second[i](y)
            splitted_x = torch.matmul(self.softmax(in_softmax), out_softmax)
            splitted_x_list.append(splitted_x)
        concatenated_x = torch.cat(splitted_x_list, dim = 2)
        multi_head_attention_output_second = self.dropout(self.w_O_second(concatenated_x))

        # Masked Multi-Head Attention
        multi_head_attention_output_second += multi_head_attention_output_first
        multi_head_attention_output_second = self.layer_normalization2(multi_head_attention_output_second)

        ### Sublayer 3
        # Feed-Forward Network
        positionwise_output_list = [] # (0, 2, 1)
        for position in range(multi_head_attention_output_second.shape[1]): # token length
            positionwise_input = multi_head_attention_output_second[:, position, :]
            positionwise_output = self.relu(self.feed_forward_network1(positionwise_input))
            positionwise_output = self.feed_forward_network2(positionwise_output)
            positionwise_output_list.append(positionwise_output)
        ffn_output = self.dropout(torch.stack(positionwise_output_list, dim = 1))

        # Add & Layer Normalization
        ffn_output += multi_head_attention_output_second
        ffn_output = self.layer_normalization3(ffn_output)
        return ffn_output

    def masking(self, x):
        masking_tensor = torch.triu(torch.empty(x.shape[1], x.shape[2]).fill_(float("-inf")), diagonal = 1)
        for idx in range(x.shape[0]):
            x[idx] += masking_tensor
        return x