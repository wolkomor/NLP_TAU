# -*- coding: utf-8 -*-

import torch

from util import ConfusionMatrix, Progbar, minibatches
from data_util import get_chunks, load_embeddings
from defs import LBLS

class TrainerBase:
    def __init__(self, model, config, helper, logger):
        self._model = model
        self._config = config
        self._helper = helper
        self._logger = logger

        self._evaluator = Evaluator(Predictor(model, config))

    def train(self, train_examples, dev_examples):
        model = self._model
        config = self._config
        logger = self._logger

        best_score = 0.

        preprocessed_train_examples = train_examples['preprocessed']
        step = 0
        for epoch in range(config.n_epochs):
            model.train()
            logger.info("Epoch %d out of %d", epoch + 1, config.n_epochs)
            prog = Progbar(target = 1 + int(len(preprocessed_train_examples) / config.batch_size))

            avg_loss = 0
            for i, minibatch in enumerate(minibatches(preprocessed_train_examples, config.batch_size)):
                sentences = torch.tensor(minibatch[0], device=config.device)
                labels = torch.tensor(minibatch[1], device=config.device)
                masks = torch.tensor(minibatch[2], device=config.device)
                avg_loss += self._train_on_batch(sentences, labels, masks)
            avg_loss /= i + 1
            logger.info("Training average loss: %.5f", avg_loss)

            model.eval()
            with torch.no_grad():
                logger.info("Evaluating on development data")
                token_cm, entity_scores = self._evaluator.evaluate(dev_examples)
                logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
                logger.debug("Token-level scores:\n" + token_cm.summary())
                logger.info("Entity level P/R/F1: {:.2f}/{:.2f}/{:.2f}".format(*entity_scores))

                score = entity_scores[-1]
                
                if score > best_score and config.model_output:
                    best_score = score
                    logger.info("New best score! Saving model in %s", config.model_output)
                    torch.save(model.state_dict(), config.model_output)
                print("")
        return best_score

    def _train_on_batch(self, sentences, labels, masks):
        raise NotImplementedError

class Predictor:
    def __init__(self, model, config):
        self._model = model
        self._config = config

    def predict(self, examples, use_str_labels=False):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        config = self._config
        preprocessed_examples = examples['preprocessed']

        preds = []
        prog = Progbar(target=1 + int(len(preprocessed_examples) / config.batch_size))
        for i, minibatch in enumerate(minibatches(preprocessed_examples, config.batch_size, shuffle=False)):
            sentences = torch.tensor(minibatch[0], device=config.device)
            tag_probs = self._model(sentences)
            preds_ = torch.argmax(tag_probs, dim=-1)
            preds += list(preds_)
            prog.update(i + 1, [])

        return self.consolidate_predictions(examples, preds, use_str_labels)

    @staticmethod
    def consolidate_predictions(examples, preds, use_str_labels=False):
        """Batch the predictions into groups of sentence length.
        """
        assert len(examples['tokens']) == len(examples['preprocessed'])
        assert len(examples['tokens']) == len(preds)

        ret = []
        for i, (sentence, labels) in enumerate(examples['token_indices'] if not use_str_labels else examples['tokens']):
            _, _, mask = examples['preprocessed'][i]
            labels_ = [l.item() for l, m in zip(preds[i], mask) if m] # only select elements of mask.
            assert len(labels_) == len(labels)
            ret.append([sentence, labels, labels_])
        return ret


class Evaluator:
    def __init__(self, predictor):
        self._predictor = predictor

    def evaluate(self, examples):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Returns:
            The F1 score for predicting tokens as named entities.
        """
        token_cm = ConfusionMatrix(labels=LBLS)

        correct_preds, total_correct, total_preds = 0., 0., 0.
        for data  in self._predictor.predict(examples):
            (_, labels, labels_) = data

            for l, l_ in zip(labels, labels_):
                token_cm.update(l, l_)
            gold = set(get_chunks(labels))
            pred = set(get_chunks(labels_))
            correct_preds += len(gold.intersection(pred))
            total_preds += len(pred)
            total_correct += len(gold)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return token_cm, (p, r, f1)

class BaseDataPreprocessor:
    def __init__(self, model, config, helper):
        self._max_length = model._max_length
        self._window_size = config.window_size
        self._n_features = config.n_features
        self._helper = helper
        
    def preprocess_sequence_data(self, examples):
        def featurize_windows(data, start, end, window_size = 1):
            """Uses the input sequences in @data to construct new windowed data points.
            """
            ret = []
            for sentence, labels in data:
                from util import window_iterator
                sentence_ = []
                for window in window_iterator(sentence, window_size, beg=start, end=end):
                    sentence_.append(sum(window, []))
                ret.append((sentence_, labels))
            return ret

        preprocessed_examples = featurize_windows(examples['token_indices'], self._helper.START, self._helper.END, self._window_size)
        examples['preprocessed'] = self.pad_sequences(preprocessed_examples)
        return examples

    def pad_sequences(self, examples):
        raise NotImplementedError

# To use as a mock
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self