###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
from typing import Tuple, Union
from pathlib import Path

import torch
import random

from language_modelling.awdlstmlm import data

SEED = 1111
TEMPERATURE = 1.0


class Generator:
    _model = None
    _model_path = None
    _corpus = None

    def __init__(self,
                 model_path: Union[str, Path],
                 corpus_path: Union[str, Path],
                 seed: int = SEED,
                 cuda: bool = True,
                 temperature: float = TEMPERATURE):
        self._device = torch.device('cuda') if cuda else torch.device('cpu')
        self._temperature = temperature
        if self._model is None or model_path != self._model_path:
            torch.cuda.manual_seed(seed) if cuda else torch.manual_seed(seed)
            self._model_path = model_path
            self._cuda = cuda
            self._model = self._load_model()
            self._corpus, self._ntokens = self._load_corpus(corpus_path)

    def _load_model(self) -> torch.nn.Module:
        with open(self._model_path, 'rb') as f:
            model, _, _ = torch.load(f, map_location=self._device)
        model.eval()
        return model

    def _load_corpus(self, corpus_path) -> Tuple[data.Corpus, int]:
        corpus = data.Corpus(corpus_path, self._cuda)
        ntokens = len(corpus.dictionary)
        return corpus, ntokens

    def predict_nis(self, segmented_data, top_k=3):
        candidates = []

        with torch.no_grad():
            input_idxs_tensor = torch.tensor([[0]]).to(self._device)
            hidden = self._model.init_hidden(input_idxs_tensor.size()[1])

            for i in range(len(segmented_data)):
                input_token = segmented_data[i]
                idx = self._corpus.dictionary.word2idx.get(input_token)
                if not idx:
                    idx = random.choice(list(self._corpus.dictionary.word2idx.values()))
                input_idxs_tensor.data.fill_(idx)
                output, hidden = self._model(input_idxs_tensor, hidden)
                if i != len(segmented_data) - 1:
                    continue
                word_weights = self._model.decoder(output).squeeze().data.exp()
                _, predictions = torch.topk(word_weights, top_k)
                for new_idx in predictions:
                    the_idx = new_idx.item()
                    candidate = self._corpus.dictionary.idx2word[the_idx]
                    candidates.append(candidate)
        return candidates

    def generate(self, input_str: str, output_len: int) -> str:
        user_input_tokens = input_str.split(",")
        result = self.predict_nis(user_input_tokens, output_len + 7)[6:]
        return ','.join(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PTB Langmodeluage Model')

    # Model parameters.
    parser.add_argument('--data', type=str, default='./data/penn',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (LSTM, QRNN)')
    parser.add_argument('--checkpoint', type=str, default='./model.pt',
                        help='model checkpoint to use')
    parser.add_argument('--outf', type=str, default='generated.txt',
                        help='output file for generated text')
    parser.add_argument('--words', type=int, default='1000',
                        help='number of words to generate')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE,
                        help='temperature - higher will increase diversity')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='reporting interval')
    parser.add_argument('--input', type=str, default='')
    args = parser.parse_args()
    generator = Generator(model_path=args.checkpoint,
                          corpus_path=args.data,
                          seed=args.seed,
                          cuda=args.cuda,
                          temperature=args.temperature)
    result = generator.generate(args.input, int(args.words))
    with open(args.outf, 'w') as f:
        f.write(result)
