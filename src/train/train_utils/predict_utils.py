import operator
from itertools import groupby

import numpy as np
import torch
from numpy.typing import NDArray

RESULT_LABELS_TYPE = list[list[int]]
RESULT_CONFIDENCES_TYPE = list[NDArray]


def matrix_to_string(model_output: torch.Tensor, vocab: str) -> tuple[list[str], list[NDArray]]:
    labels, confs = postprocess(model_output)
    labels_decoded, conf_decoded = decode(labels_raw=labels, conf_raw=confs)
    string_pred = labels_to_strings(labels_decoded, vocab)
    return string_pred, conf_decoded


def postprocess(model_output: torch.Tensor) -> tuple[NDArray, NDArray]:
    output = model_output.permute(1, 0, 2)
    output = torch.nn.Softmax(dim=2)(output)
    confidences, labels = output.max(dim=2)
    confidences = confidences.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    return labels, confidences


# TODO: split the function
def decode(  # noqa: WPS210
    labels_raw: NDArray,
    conf_raw: NDArray,
) -> tuple[RESULT_LABELS_TYPE, RESULT_CONFIDENCES_TYPE]:
    result_labels = []
    result_confidences = []
    for label, conf in zip(labels_raw, conf_raw):
        result_one_labels = []
        result_one_confidences = []
        for result_label, group in groupby(zip(label, conf), operator.itemgetter(0)):
            if result_label > 0:
                result_one_labels.append(result_label)
                result_one_confidences.append(max(list(zip(*group))[1]))
        result_labels.append(result_one_labels)
        result_confidences.append(np.array(result_one_confidences))

    return result_labels, result_confidences


def labels_to_strings(labels: RESULT_LABELS_TYPE, vocab: str) -> list[str]:
    strings = []
    for single_str_labels in labels:
        try:
            output_str = _get_output_str(single_str_labels, vocab)
        except IndexError:
            strings.append('Error')
        else:
            strings.append(output_str)
    return strings


def _get_output_str(single_str_labels: list[int], vocab: str) -> str:
    return ''.join(vocab[char_index - 1] if char_index > 0 else '_' for char_index in single_str_labels)  # noqa: WPS221
