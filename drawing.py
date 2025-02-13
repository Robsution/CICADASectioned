import hls4ml
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import numpy.typing as npt

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from typing import List


class Draw:
    def __init__(self, output_dir: Path = Path("plots")):
        self.output_dir = output_dir
        self.cmap = ["green", "red", "blue", "orange", "purple", "brown"]
        hep.style.use("CMS")

    def _parse_name(self, name: str) -> str:
        return name.replace(" ", "-").lower()

    def plot_loss_history(
        self, training_loss: npt.NDArray, validation_loss: npt.NDArray, name: str
    ):
        plt.plot(np.arange(1, len(training_loss) + 1), training_loss, label="Training")
        plt.plot(
            np.arange(1, len(validation_loss) + 1), validation_loss, label="Validation"
        )
        plt.legend(loc="upper right")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(
            f"{self.output_dir}/{self._parse_name(name)}.png", bbox_inches="tight"
        )
        plt.close()

    def plot_regional_deposits(self, deposits: npt.NDArray, mean: float, name: str):
        im = plt.imshow(
            deposits.reshape(6, 14), vmin=0, vmax=deposits.max(), cmap="Purples"
        )
        ax = plt.gca()
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(r"Calorimeter E$_T$ deposit (GeV)")
        plt.xticks(np.arange(14), labels=np.arange(4, 18))
        plt.yticks(
            np.arange(6),
            labels=np.arange(6)[::-1],
            rotation=90,
            va="center",
        )
        plt.xlabel(r"i$\eta$")
        plt.ylabel(r"i$\phi$")
        plt.title(rf"Mean E$_T$ {mean: .2f} ({name})")
        plt.savefig(
            f"{self.output_dir}/profiling-mean-deposits-{self._parse_name(name)}.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_spacial_deposits_distribution(
        self, deposits: List[npt.NDArray], labels: List[str], name: str
    ):
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        for deposit, label in zip(deposits, labels):
            bins = np.argwhere(deposit)
            phi, eta = bins[:, 1], bins[:, 2]
            ax1.hist(
                eta + 4,
                density=True,
                facecolor=None,
                bins=np.arange(4, 19),
                label=label,
                histtype="step",
            )
            ax2.hist(
                phi,
                density=True,
                facecolor=None,
                bins=np.arange(19),
                label=label,
                histtype="step",
            )
        ax1.set_ylabel("a.u.")
        ax1.set_xlabel(r"i$\eta$")
        ax2.set_xlabel(r"i$\phi$")
        plt.legend(loc="best")
        plt.savefig(
            f"{self.output_dir}/profiling-spacial-{self._parse_name(name)}.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_deposits_distribution(
        self, deposits: List[npt.NDArray], labels: List[str], name: str
    ):
        for deposit, label in zip(deposits, labels):
            plt.hist(
                deposit.reshape((-1)),
                bins=100,
                range=(0, 1024),
                density=1,
                label=label,
                log=True,
                histtype="step",
            )
        plt.xlabel(r"E$_T$")
        plt.legend(loc="best")
        plt.savefig(
            f"{self.output_dir}/profiling-deposits-{self._parse_name(name)}.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_reconstruction_results(
        self,
        deposits_in: npt.NDArray,
        deposits_out: npt.NDArray,
        loss: float,
        name: str,
    ):
        fig, (ax1, ax2, ax3, cax) = plt.subplots(
            ncols=4, figsize=(15, 10), gridspec_kw={"width_ratios": [1, 1, 1, 0.05]}
        )
        max_deposit = max(deposits_in.max(), deposits_out.max())

        ax1 = plt.subplot(1, 4, 1)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.set_title("Original", fontsize=18)
        ax1.imshow(
            deposits_in.reshape(6, 14), vmin=0, vmax=max_deposit, cmap="Purples"
        )

        ax2 = plt.subplot(1, 4, 2)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.set_title("Reconstructed", fontsize=18)
        ax2.imshow(
            deposits_out.reshape(6, 14), vmin=0, vmax=max_deposit, cmap="Purples"
        )

        ax3 = plt.subplot(1, 4, 3)
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax3.set_title(rf"|$\Delta$|, MSE: {loss: .2f}", fontsize=18)

        im = ax3.imshow(
            np.abs(deposits_in - deposits_out).reshape(6, 14),
            vmin=0,
            vmax=max_deposit,
            cmap="Purples",
        )

        ip = InsetPosition(ax3, [1.05, 0, 0.05, 1])
        cax.set_axes_locator(ip)
        fig.colorbar(im, cax=cax, ax=[ax1, ax2, ax3]).set_label(
            label=r"Calorimeter E$_T$ deposit (GeV)", fontsize=18
        )

        plt.savefig(
            f"{self.output_dir}/reconstruction-results-{self._parse_name(name)}.png", bbox_inches="tight"
        )
        plt.close()

    def plot_anomaly_score_distribution(
        self, scores: List[npt.NDArray], labels: List[str], name: str
    ):
        for score, label in zip(scores, labels):
            plt.hist(
                score.reshape((-1)),
                bins=100,
                range=(0, 256),
                density=1,
                label=label,
                log=True,
                histtype="step",
            )
        plt.xlabel(r"Anomaly Score")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(
            f"{self.output_dir}/score-dist-{self._parse_name(name)}.png", bbox_inches="tight"
        )
        plt.close()

    def plot_roc_curve(
        self,
        y_trues: List[npt.NDArray],
        y_preds: List[npt.NDArray],
        labels: List[str],
        inputs: List[npt.NDArray],
        name: str,
        cv: int = 3,
    ):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        for y_true, y_pred, label, color, d in zip(
            y_trues, y_preds, labels, self.cmap, inputs
        ):
            aucs = []
            for _, indices in skf.split(y_pred, y_true):
                fpr, tpr, _ = roc_curve(y_true[indices], y_pred[indices])
                aucs.append(auc(fpr, tpr))
            std_auc = np.std(aucs)

            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            fpr_base, tpr_base, _ = roc_curve(y_true, np.mean(d**2, axis=(1, 2)))

            plt.plot(
                fpr * 28.61,
                tpr,
                linestyle="-",
                lw=1.5,
                color=color,
                alpha=0.8,
                label=rf"{label} (AUC = {roc_auc: .4f} $\pm$ {std_auc: .4f})",
            )

            plt.plot(
                fpr_base * 28.61,
                tpr_base,
                linestyle="--",
                lw=1.0,
                color=color,
                alpha=0.5,
                label=rf"{label}, Baseline",
            )

        plt.plot(
            [0.003, 0.003],
            [0, 1],
            linestyle="--",
            lw=1,
            color="black",
            label="3 kHz",
        )
        plt.xlim([0.0002861, 28.61])
        plt.ylim([0.01, 1.0])
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Trigger Rate (MHz)")
        plt.ylabel("Signal Efficiency")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(
            f"{self.output_dir}/roc-{self._parse_name(name)}.png", bbox_inches="tight"
        )
        plt.close()

    def plot_compilation_error(
        self, scores_keras: npt.NDArray, scores_hls4ml: npt.NDArray, name: str
    ):
        plt.scatter(scores_keras, np.abs(scores_keras - scores_hls4ml), s=1)
        plt.xlabel("Anomaly Score, $S$")
        plt.ylabel("Error, $|S_{Keras} - S_{hls4ml}|$")
        plt.savefig(
            f"{self.output_dir}/compilation-error-{self._parse_name(name)}.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_compilation_error_distribution(
        self, scores_keras: npt.NDArray, scores_hls4ml: npt.NDArray, name: str
    ):
        plt.hist(scores_keras - scores_hls4ml, fc="none", histtype="step", bins=100)
        plt.xlabel("Error, $S_{Keras} - S_{hls4ml}$")
        plt.ylabel("Number of samples")
        plt.yscale("log")
        plt.savefig(
            f"{self.output_dir}/compilation-error-dist-{self._parse_name(name)}.png",
            bbox_inches="tight",
        )

    def plot_cpp_model(self, hls_model, name: str):
        hls4ml.utils.plot_model(
            hls_model,
            show_shapes=True,
            show_precision=True,
            to_file=f"{self.output_dir}/cpp-model-{self._parse_name(name)}.png",
        )

    def plot_roc_curve_comparison(
        self, scores_keras: dict, scores_hls4ml: npt.NDArray, name: str
    ):
        fpr_model: list = []
        tpr_model: list = []

        scores_keras_normal = scores_keras["Background"]
        scores_hls4ml_normal = scores_hls4ml["Background"]

        for dataset_name, color in zip(list(scores_keras.keys())[:-1], self.cmap):
            scores_keras_anomaly = scores_keras[dataset_name]
            scores_hls4ml_anomaly = scores_hls4ml[dataset_name]

            y_true = np.append(
                np.zeros(len(scores_keras_normal)), np.ones(len(scores_hls4ml_anomaly))
            )
            y_score_keras = np.append(scores_keras_normal, scores_keras_anomaly)
            y_score_hls = np.append(scores_hls4ml_normal, scores_hls4ml_anomaly)

            for y_scores, model, ls in zip(
                [y_score_keras, y_score_hls], ["Keras", "hls4ml"], ["-", "--"]
            ):
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                plt.plot(
                    fpr * 28.61,
                    tpr,
                    linestyle=ls,
                    color=color,
                    label="{0}: {1}, AUC = {2:.4f}".format(
                        model, dataset_name, auc(fpr, tpr)
                    ),
                )

        plt.plot(
            [0.003, 0.003],
            [0, 1],
            linestyle="--",
            color="black",
            label="3 kHz trigger rate",
        )
        plt.xlim([0.0002861, 28.61])
        plt.ylim([0.01, 1.0])
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Trigger Rate (MHz)")
        plt.ylabel("Signal Efficiency")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(
            f"{self.output_dir}/compilation-roc-{self._parse_name(name)}.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_output_reference(self):
        with open("misc/output-reference.txt") as f:
            data = f.read()
        data = np.array([row.split(",") for row in data.split("\n")[:-1]]).astype(
            np.int8
        )
        data = np.flipud(data) - 1
        legend_elements = [
            Patch(
                facecolor=self.cmap[0],
                edgecolor=self.cmap[0],
                label="Anomaly Detection, Integer Part",
            ),
            Patch(
                facecolor=self.cmap[1],
                edgecolor=self.cmap[1],
                label="Anomaly Detection, Decimal Part",
            ),
            Patch(
                facecolor=self.cmap[2], edgecolor=self.cmap[2], label="Heavy Ion Bit"
            ),
            Patch(facecolor=self.cmap[3], edgecolor=self.cmap[3], label="Reserved"),
        ]
        plt.figure(figsize=(25, 5))
        plt.pcolor(
            data, edgecolors="black", alpha=0.6, cmap=ListedColormap(self.cmap[:4])
        )
        plt.xticks([])
        plt.yticks([])
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                plt.text(
                    x + 0.5,
                    y + 0.5,
                    abs(y * 32 + x - 191),
                    horizontalalignment="center",
                    fontsize=16,
                    verticalalignment="center",
                )
        plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(0, 0),
            loc="upper left",
            frameon=False,
            ncol=4,
            borderaxespad=0,
        )
        plt.savefig(
            f"{self.output_dir}/ugt-link-reference.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_results_supervised(
        self, grid: npt.NDArray, models: list[str], datasets: list[str], name: str
    ):
        plt.imshow(grid, alpha=0.7, cmap="RdYlGn")
        plt.xticks(
            np.arange(len(models)),
            labels=models,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        plt.yticks(np.arange(len(datasets)), labels=datasets)
        for i in range(len(datasets)):
            for j in range(len(models)):
                text = plt.text(
                    j,
                    i,
                    "{0:.3f}".format(grid[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    size=16,
                )
        plt.savefig(
            f"{self.output_dir}/supervised-{self._parse_name(name)}.png",
            bbox_inches="tight",
        )
        plt.close()
