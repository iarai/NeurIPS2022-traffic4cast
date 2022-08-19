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
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# TODO print non-classified timesteps as well and what the model predicts there
# TODO print daytime in x axis labels
def plot_segment_classifications_simple(  # noqa:C901
    cc_pred, cc_true, dayline_labels=None, attention_timestamps=None, losses=None, proba_green=None, proba_yellow=None, proba_red=None
):
    fig, axs = plt.subplots(4, figsize=(50, 30), tight_layout=True, sharex=True)

    if dayline_labels is None:
        dayline_labels = range(len(cc_pred))

    def pl(ax, classifications, other_classifications=None):
        ax.plot(dayline_labels, classifications, linestyle="", marker="_")
        for t in range(len(classifications)):
            cc = classifications[t]

            if cc == 0:
                ax.add_patch(Rectangle((t, 0), 1, 1, fill=True, facecolor="grey"))
            elif cc == 1:
                ax.add_patch(Rectangle((t, 1), 1, 1, fill=True, facecolor="green"))
            elif cc == 2:
                ax.add_patch(Rectangle((t, 2), 1, 1, fill=True, facecolor="yellow"))
            elif cc == 3:
                ax.add_patch(Rectangle((t, 3), 1, 1, fill=True, facecolor="red"))
            if other_classifications is not None:
                other_cc = other_classifications[t]
                if other_cc == cc:
                    continue
                if other_cc == 0:
                    ax.add_patch(Rectangle((t, 0), 1, 1, fill=True, hatch="x", facecolor="lightgrey"))
                elif other_cc == 1:
                    ax.add_patch(Rectangle((t, 1), 1, 1, fill=True, hatch="x", facecolor="lightgreen"))
                elif other_cc == 2:
                    ax.add_patch(Rectangle((t, 2), 1, 1, fill=True, hatch="x", facecolor="khaki"))
                elif other_cc == 3:
                    ax.add_patch(Rectangle((t, 3), 1, 1, fill=True, hatch="x", facecolor="mistyrose"))
        ax.set_ylim([0, 4])

    ax = axs[0]
    pl(ax, cc_true)
    ax.set_title("cc_true")
    ax.grid()
    # TODO derive from cc_pred/cc_true
    if attention_timestamps is not None:
        ax.scatter([dayline_labels[l] for l in attention_timestamps], [0] * len(attention_timestamps), marker="x", color="red", s=500)
    ax = axs[1]
    pl(ax, cc_pred, cc_true)
    ax.set_title("cc_pred")
    ax.grid()

    ax = axs[2]
    if losses is not None:
        ax.bar([i + 0.5 for i in range(len(dayline_labels))], losses, width=0.2)
        ax.set_ylim([0, 7])
        ax.grid()
    axs[2].set_title("loss")

    ax = axs[3]
    if proba_green is not None:
        ax.plot([i + 0.5 for i in range(len(dayline_labels))], proba_green, color="green")
    if proba_yellow is not None:
        ax.plot([i + 0.5 for i in range(len(dayline_labels))], proba_yellow, color="yellow")
    if proba_red is not None:
        ax.plot([i + 0.5 for i in range(len(dayline_labels))], proba_red, color="red")
    ax.set_ylim([0, 1])
    axs[3].set_title("proba_pred")
    axs[3].grid()
    return fig, axs
