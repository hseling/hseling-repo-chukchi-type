import tempfile
import pickle
import torch
import numpy
from typing import Dict, Iterable, List, Optional, Union

from embeddings.iiksiin.autoencoder import (
    Autoencoder, Tensors as AutoencoderTensors, UnbindingLoss)
from embeddings.iiksiin.corpus2tensors import main as c2t_main
from embeddings.iiksiin.iiksiin import Alphabet

MORPHEME_DELIMITER: str = '>'
END_OF_MORPHEME_SYMBOL: str = '\\u0000'
PADDING_SYMBOL: str = '\\u0004'
BLACKLIST_CHAR: str = '*'


class Iiksiin:
    def __init__(self, corpus: Iterable[str]):
        self.corpus: Iterable[str] = corpus
        self.alphabet: Optional[Alphabet] = None
        self.tensors: Optional[Dict[str, torch.Tensor]] = None
        self._data: Optional[AutoencoderTensors] = None
        self._batch_size: Optional[int] = 100
        self.device: int = 0 if torch.cuda.is_available() else -1
        self.model: Optional[Autoencoder] = None
        self.vectors: Optional[Dict[str, List[float]]] = None
        self.embeddings = None

    @staticmethod
    def _wrap_symbol(symbol: str) -> str:
        return str.encode(symbol).decode("unicode_escape")

    def generate_alphabet(self, name: str, input_source: Iterable[str]):
        self.alphabet = Alphabet.create_from_source(
            name, input_source,
            self._wrap_symbol(MORPHEME_DELIMITER),
            self._wrap_symbol(END_OF_MORPHEME_SYMBOL),
            PADDING_SYMBOL, BLACKLIST_CHAR)

    def generate_tensors(
            self, max_characters: int = 20, max_morphemes: int = 10):
        with tempfile.NamedTemporaryFile('wt') as input_file,\
                tempfile.NamedTemporaryFile('wb+') as output_file:
            input_file.write('\n'.join(self.corpus))
            input_fname = input_file.name
            output_fname = output_file.name
            self.tensors = c2t_main(
                max_characters, max_morphemes, self.alphabet,
                self._wrap_symbol(END_OF_MORPHEME_SYMBOL),
                self._wrap_symbol(MORPHEME_DELIMITER),
                input_fname, output_fname, 0, BLACKLIST_CHAR)

    def init_model(self, hidden_layer_size: int = 50, hidden_layers: int = 2):
        self._data = AutoencoderTensors(self.tensors, self.alphabet)
        self.model = Autoencoder(
            input_dimension_size=self._data.input_dimension_size,
            hidden_layer_size=hidden_layer_size,
            num_hidden_layers=hidden_layers,
        )

    def train_model(
            self,
            cuda_device: int = -1,
            save_frequency: int = 100,
            epochs: int = 200,
            batch_size: int = 100,
            learning_rate: float = 0.01,
            output_path: Optional[str] = None):
        device = "cpu" if cuda_device == -1 else f"cuda:{cuda_device}"
        criterion: torch.nn.Module = \
            UnbindingLoss(alphabet=self._data.alphabet).to(device)
        optimizer: torch.optim.Optimizer = \
            torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self._batch_size = batch_size
        self.device = cuda_device
        self.model.run_training(
            self._data, criterion, optimizer,
            epochs, batch_size, save_frequency,
            cuda_device,
        )
        if output_path:
            torch.save(self.model, output_path)

    def generate_vectors(self):
        self.vectors = \
            self.model.run_t2v(self._data, self._batch_size, self.device)

    def embed(self):
        already_taken = {}
        unique = 0
        res = []
        for sent in self.corpus:
            for word in sent.split():
                for segm in word.split(">"):
                    if len(segm) == 0:
                        continue
                    try:
                        _ = already_taken[segm]
                        continue
                    except KeyError:
                        try:
                            vect = self.vectors[segm]
                            res.append(vect)
                        except KeyError:
                            unique += 1
                            vect = numpy.array([0.0] * 64)
                            for char in list(segm):
                                try:
                                    c_vect = numpy.array(self.vectors[char])
                                    vect += c_vect
                                except KeyError:
                                    continue
                            vect /= len(list(segm))
                            res.append(vect.tolist())
                        already_taken[segm] = True
        self.embeddings = res
        return unique

    def run(self, model_output_path=None, vectors_output_path=None):
        """Run pipeline with default values"""
        self.generate_alphabet('alphabet', self.corpus)
        self.generate_tensors()
        self.init_model()
        self.train_model(output_path=model_output_path)
        self.generate_vectors()
        self.embed()
        if vectors_output_path:
            with open(vectors_output_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
