import os
import re
import subprocess
from pathlib import Path
from collections import Counter
import pandas as pd
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from string import punctuation


def process_line(line):
    return "".join(ch.lower() + " S-MORPH\n" if ch != " " else "\n\n" for ch in line if ch not in punctuation)


def label(word):
    B = "B-MORPH"
    M = "M-MORPH"
    E = "E-MORPH"
    S = "S-MORPH"

    target = ""
    morphemes = word.split(">")
    for morph in morphemes:
        if morph:
            morph_len = len(morph)
            if morph_len == 1:
                target += f"{morph} {S}\n"
            else:
                target += f"{morph[0]} {B}\n"
                for ch_idx in range(1, morph_len - 1):
                    target += f"{morph[ch_idx]} {M}\n"
                target += f"{morph[-1]} {E}\n"
    return target


class NCRFpp(object):
    def __init__(self, corpus_home, corpus_name, out_path, n_splits):
        self.corpus_home = Path(corpus_home)
        self.corpus_name = corpus_name
        self.out_folder = Path(out_path)
        self.out_folder.mkdir(exist_ok=True)
        self.cwd = Path.cwd()
        self.n_splits = n_splits

        self.config_template = ""
        self.decode_config_template = ""

        self.sentences = []

        self._make_config_templates()
        self._fill_corpus_data()

    def _make_config_templates(self):
        self.config_template = '''train_dir={train_path}
        dev_dir={test_path}
        test_dir={test_path}
        model_dir={model_path}
        log_dir={log_path}
        seg=True
        char_emb_dim=30
        use_crf=True
        use_char=True
        word_seq_feature=CNN
        char_seq_feature=CNN
        nbest=1
        status=train
        optimizer=SGD
        iteration=1000
        batch_size=10
        ave_batch_loss=False
        cnn_layer=4
        char_hidden_dim=50
        dropout=0.5
        lstm_layer=0
        bilstm=False
        learning_rate=0.015
        lr_decay=0.05
        momentum=0
        l2=1e-8
        gpu=True'''

        self.decode_config_template = '''status=decode
        raw_dir={raw_dir}
        nbest=1
        decode_dir={decode_dir}
        dset_dir={dset_dir}
        load_model_dir={model_dir}'''

    def _fill_corpus_data(self):
        with open(self.corpus_home.joinpath(self.corpus_name)) as corpus_file:
            self.sentences = [line.rstrip() for line in corpus_file]

    def data_for_learning(self):
        tokens = [word for line in self.sentences for word in line.split()]
        types = Counter(tokens)
        morphemes = [morph for word in types for morph in word.split(">")]
        morph_stats = Counter(morphemes)
        rare_morphemes = {morph for morph in morph_stats if morph_stats[morph] < 2}
        representative_words = [word for word in types if not set(word.split(">")).intersection(rare_morphemes)]
        words_df = pd.DataFrame(representative_words, columns=["word"])
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        self.make_fractions(kfold, words_df)

    # делаем несколько фолдов:
    # !ls -la segmentation
    # ---------------------------
    # drwxr-xr-x 12 root root 4096 Oct  6 08:40 .
    # drwxr-xr-x  1 root root 4096 Oct  6 08:40 ..
    # drwxr-xr-x  2 root root 4096 Oct  6 08:40 fold_1
    # drwxr-xr-x  2 root root 4096 Oct  6 08:40 fold_10
    # drwxr-xr-x  2 root root 4096 Oct  6 08:40 fold_2
    # drwxr-xr-x  2 root root 4096 Oct  6 08:40 fold_3
    # drwxr-xr-x  2 root root 4096 Oct  6 08:40 fold_4
    # drwxr-xr-x  2 root root 4096 Oct  6 08:40 fold_5
    # drwxr-xr-x  2 root root 4096 Oct  6 08:40 fold_6
    # drwxr-xr-x  2 root root 4096 Oct  6 08:40 fold_7
    # drwxr-xr-x  2 root root 4096 Oct  6 08:40 fold_8
    # drwxr-xr-x  2 root root 4096 Oct  6 08:40 fold_9

    # будем учить каждый из фолдов, чтобы выбрать лучшую модель
    def make_fractions(self, kfold, words_df):
        for idx, (train, test) in enumerate(kfold.split(words_df.index)):
            idx += 1
            train_words = words_df.loc[train].word
            test_words = words_df.loc[test].word
            fold_path = self.out_folder.joinpath(f"fold_{idx}")
            fold_path.mkdir(parents=True, exist_ok=True)
            train_path = fold_path.joinpath("train.bmes")
            test_path = fold_path.joinpath("test.bmes")
            config_path = fold_path.joinpath("hparams.config")
            model_path = fold_path.joinpath("model")
            log_path = fold_path.joinpath("training.log")
            with open(config_path, "w") as config_file:
                config_file.write(self.config_template.format(train_path=train_path.absolute(),
                                                         test_path=test_path.absolute(),
                                                         model_path=model_path.absolute(),
                                                         log_path=log_path.absolute()))
            for fraction_path, fraction in zip((train_path, test_path), (train_words, test_words)):
                with open(fraction_path, "w") as file:
                    file.writelines(label(word) + "\n" for word in fraction)

    def make_raw(self, data_to_segment, corpus_raw_name):
        with open(self.corpus_home.joinpath(data_to_segment)) as txt_corpus, \
                open(self.corpus_home.joinpath(corpus_raw_name), "w") as raw_file:
            for line in txt_corpus:
                raw_file.write(process_line(line.strip()) + "\n\n.\n\n")

    # обучение одного фолда
    def fit(self, fold_number):
        # в ноутбуке здесь "!python ncrfpp_project/main.py --config segmentation/fold_1/hparams.config > segmentation/fold_1/log.log"
        # то есть вызываем для конфига self.cwd/fold_i/hparams.config (fold_i = "fold" + fold_number)
        # для логов self.cwd /fold_i/log.log
        # из логов потом найдем номер эпохи, на которой была лучшая метрика
        pass

    def find_best_epoch(self, fold_number):
        # fold_i = "fold" + fold_number
        # !cat self.cwd/fold_i/log.log | grep 'best'
        # -----------------------
        # Exceed previous best f score: -10
        # Save current best model in file: /content/segmentation/fold_1/model.0.model
        # Exceed previous best f score: 0.3469785575048733
        # Save current best model in file: /content/segmentation/fold_1/model.1.model
        # Exceed previous best f score: 0.3651626442812172
        # Save current best model in file: /content/segmentation/fold_1/model.2.model
        # Exceed previous best f score: 0.4556451612903226

        # очевидно, берем последнее
        pass

    # запуск модели делается с использованием конфига, из которого читаются нужные параметры
    # так что загрузить старую модель = написать корректный конфиг
    def load_model(self, model_path, dset_path, segmented_corpus_path, decode_config_path, corpus_raw_name):
        Path(self.out_folder.joinpath(segmented_corpus_path).absolute()).touch()
        config_params = {"model_dir": self.out_folder.joinpath(model_path).absolute(),
                         "dset_dir": self.out_folder.joinpath(dset_path).absolute(),
                         "decode_dir": self.out_folder.joinpath(segmented_corpus_path).absolute(),
                         "raw_dir": self.corpus_home.joinpath(corpus_raw_name).absolute()}
        with open(self.out_folder.joinpath(decode_config_path).absolute(), "w") as config_file:
            config_file.write(self.decode_config_template.format(**config_params))

    # создает в decode_dir(см. выше) файл segmented_corpus_path формата .bmes
    def decode(self, PYTHON, ROOT, decode_config_path):
        # в ноутбуке здесь "!python ncrfpp_project/main.py --config self.out_folder/decode_config_path"
        segmentor_proc = subprocess.run(
            f'{PYTHON} {ROOT}/segmentation/ncrfpp_project/main.py --config {self.out_folder}/{decode_config_path}',
            shell=True, capture_output=True)
        if segmentor_proc.returncode != 0:
            print("FUCK IT")

    def delete_corpus_files(self, *args):
        for arg in args:
            os.remove(self.corpus_home.joinpath(arg))

    def delete_results_files(self, *args):
        for arg in args:
            os.remove(self.out_folder.joinpath(arg))

    # делаем из bmes слова из сегментов
    def convert_bmes_to_words(self, segmented_corpus_path, segmented_corpus_words_path):
        segmented_words = []
        with open(self.out_folder.joinpath(segmented_corpus_path).absolute()) as f:
            word = ''
            for l in f.readlines():
                if re.match('^[\s\n]*$', l) and word != '':
                    segmented_words.append(word.strip(">").replace(">>", ">") + "\n")
                    word = ''
                    continue
                try:
                    ch, typ = l.split()
                    if typ == "B-MORPH":
                        word += ">" + ch
                    if typ == "E-MORPH":
                        word += ch + ">"
                    if typ == "S-MORPH":
                        word += ">" + ch + ">"
                    if typ == "M-MORPH":
                        word += ch
                except ValueError:
                    if word != '':
                        segmented_words.append(word.strip(">").replace(">>", ">") + "\n")
                        word = ''
                    continue
            if word != '':
                segmented_words.append(word.strip(">").replace(">>", ">") + "\n")
        with open(self.corpus_home.joinpath(segmented_corpus_words_path), "w") as f:
            f.writelines(segmented_words)

    def convert_words_to_strings(self, original_text, segmented_corpus_words_path):
        with open(self.corpus_home.joinpath(original_text)) as corpus_file:
            original = corpus_file.read()
        original = original.replace('—', '')
        original = original.replace('>', '')
        original = original.split('\n')
        sent2word = []
        for sent in original:
            sent2word.append(sent.split())
        with open(self.corpus_home.joinpath(segmented_corpus_words_path)) as corpus_file:
            segmented = corpus_file.read()
        segmented = segmented.replace('—', '')
        segmented = segmented.split('\n')
        segmented_sents = []
        i = 0
        for sent in sent2word:
            j = 0
            sentence = ''
            while j < len(sent):
                if j != len(sent) - 1:
                    sentence += segmented[i] + ' '
                else:
                    sentence += segmented[i]
                j += 1
                i += 1
            segmented_sents.append(sentence)
        return segmented_sents

    # делаем из слов с сегментами предложения, сразу train и test
    # код дашин, если что :о)
    def convert_words_to_sents(self, segmented_corpus_words_path, train_sents_path, test_sents_path, val_sents_path):
        with open(self.corpus_home.joinpath(self.corpus_name)) as corpus_file:
            original = corpus_file.read()
        original = original.replace('—', '')
        original = original.replace('>', '')
        original = original.split('\n')
        sent2word = []
        for sent in original:
            sent2word.append(sent.split())
        with open(segmented_corpus_words_path) as corpus_file:
            segmented = corpus_file.read()
        segmented = segmented.replace('—', '')
        segmented = segmented.split('\n')
        segmented_sents = []
        i = 0
        for sent in sent2word:
            j = 0
            sentence = ''
            while j < len(sent):
                if j != len(sent) - 1:
                    sentence += segmented[i] + ' '
                else:
                    sentence += segmented[i]
                j += 1
                i += 1
            segmented_sents.append(sentence + "\n")
        X_train, X_test = train_test_split(segmented_sents, test_size=0.2, random_state=1)

        X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=1)
        with open(train_sents_path, "w") as f:
            f.writelines(X_train)
        with open(test_sents_path, "w") as f:
            f.writelines(X_test)
        with open(val_sents_path, "w") as f:
            f.writelines(X_val)


# m = ncrfpp_project("corpus", "corpus_v5.txt", "res", 10)
# m.convert_bmes_to_words("corpus_v5_segmented.bmes", "res/corpus_v5_words.txt")
# m.convert_words_to_sents("res/corpus_v5_words.txt", "res/train.txt", "res/test.txt", "res/val.txt")
