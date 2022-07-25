#  Copyright 2022 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from t4c22.metric.masked_crossentropy import get_weights_from_class_counts
from t4c22.metric.masked_crossentropy import get_weights_from_class_fractions

# +


def _masked_crossentropy(y_hat: Tensor, y: Tensor, reduction="mean", weight: Optional[Tensor] = None, logit=True, logit_epsilon=0.00000001):
    loss = torch.nn.CrossEntropyLoss(reduction=reduction, weight=weight)
    mask = torch.isnan(y)
    y_masked = y[~mask].long()
    y_hat_masked = y_hat[~mask]

    if not logit:
        # prevent infty if y_hat is Dirac.
        y_hat_masked = torch.log(y_hat_masked + logit_epsilon)

    output = loss(y_hat_masked, y_masked)

    return output, len(y_masked)


# TODO compare with ignore_index=...
def test_masked_crossentropy():
    # in-notebook unit test
    loss = torch.nn.CrossEntropyLoss()
    # must be logits, not probs in y_hat!

    # get the base value for perfect prediction for 3 classes
    y = torch.tensor([0])
    y_hat = torch.tensor([[np.log(1), np.log(0), np.log(0)]]).float()
    assert np.isclose(loss(y_hat, y).numpy(), 0.0), loss(y_hat, y).numpy()

    y_hat = torch.tensor([[1, 0, 0]]).float()
    assert np.isclose(_masked_crossentropy(y_hat, y, logit=False)[0].numpy(), 0.0), _masked_crossentropy(y_hat, y, logit=False)[0].numpy()

    # get the base value for the uniform prediction for 3 classes
    y = torch.tensor([0])
    y_hat = torch.tensor([[np.log(1 / 3), np.log(1 / 3), np.log(1 / 3)]]).float()
    assert np.isclose(loss(y_hat, y).numpy(), 1.0986123), loss(y_hat, y).numpy()

    y = torch.tensor([2])
    y_hat = torch.tensor([[np.log(1 / 3), np.log(1 / 3), np.log(1 / 3)]]).float()
    assert np.isclose(loss(y_hat, y).numpy(), 1.0986123), loss(y_hat, y).numpy()

    # check CrossEntropyLoss does normalization
    y = torch.tensor([2])
    y_hat = torch.tensor([[np.log(1), np.log(1), np.log(1)]]).float()
    assert np.isclose(loss(y_hat, y).numpy(), 1.0986123), loss(y_hat, y).numpy()

    # check NaN in ground truth is ignored by masked_crossentropy
    y = torch.tensor([0, np.nan])
    y_hat = torch.tensor([[np.log(1), np.log(0), np.log(0)], [np.log(1 / 3), np.log(1 / 3), np.log(1 / 3)]]).float()
    assert np.isclose(_masked_crossentropy(y_hat, y)[0].numpy(), 0.0), _masked_crossentropy(y_hat, y)[0].numpy()
    assert _masked_crossentropy(y_hat, y)[1] == 1

    # check no NaN in ground truth works correctly
    y = torch.tensor([0, 1])
    y_hat = torch.tensor([[np.log(1), np.log(0), np.log(0)], [np.log(1 / 3), np.log(1 / 3), np.log(1 / 3)]]).float()
    assert np.isclose(_masked_crossentropy(y_hat, y)[0].numpy(), (0.0 + 1.0986123) / 2), loss(y_hat, y).numpy()
    assert _masked_crossentropy(y_hat, y)[1] == 2

    # check logit=False with Dirac does not produce infty.
    assert torch.isnan(_masked_crossentropy(torch.tensor([[0, 1, 0]]), torch.tensor([0.0]), logit=False)[0]).sum() == 0


def test_get_weights_from_class_nums():

    num_classes = 3
    num_samples = 22

    input = torch.randn(num_samples, num_classes, requires_grad=False)
    target = torch.empty(num_samples, dtype=torch.long).random_(num_classes)

    class_counts = np.array([(target == c).sum() for c in range(num_classes)])

    w = get_weights_from_class_counts(class_counts)

    loss = torch.nn.CrossEntropyLoss()
    class_losses = [loss(input[(target == c)], target[target == c]) for c in range(num_classes)]
    class_losses_mean = np.mean(class_losses)

    loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(w).float())
    weighted_loss = loss(input, target)
    assert np.isclose(class_losses_mean, weighted_loss), (class_losses_mean, weighted_loss)

    loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1 / c for c in class_counts]).float())
    weighted_loss = loss(input, target)
    assert np.isclose(class_losses_mean, weighted_loss), (class_losses_mean, weighted_loss)

    loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(get_weights_from_class_fractions([c / num_samples for c in class_counts])).float())
    weighted_loss = loss(input, target)
    assert np.isclose(class_losses_mean, weighted_loss), (class_losses_mean, weighted_loss)
