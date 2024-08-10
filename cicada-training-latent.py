import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import time
import numpy as np
import numpy.typing as npt
import pandas as pd
import qkeras
import tensorflow as tf
import yaml
import keras_tuner as kt

from drawing import Draw
from generator import RegionETGenerator
from models import TeacherAutoencoder, TeacherScnAutoencoder, CicadaV1, CicadaV2
from pathlib import Path
from tensorflow import data
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from typing import List
from utils import IsValidFile
from scipy.stats import wasserstein_distance
#from huggingface_hub import from_pretrained_keras

from qkeras import *
from tqdm import tqdm

INPUT_SIZE = 84

def loss(y_true: npt.NDArray, y_pred: npt.NDArray) -> npt.NDArray:
    return np.mean((y_true - y_pred) ** 2, axis=(1, 2, 3))


def quantize(arr: npt.NDArray, precision: tuple = (16, 8)) -> npt.NDArray:
    word, int_ = precision
    decimal = word - int_
    step = 1 / 2**decimal
    max_ = 2**int_ - step
    arrq = step * np.round(arr / step)
    arrc = np.clip(arrq, 0, max_)
    return arrc


def get_student_targets(
    teacher: Model, gen: RegionETGenerator, X: npt.NDArray
) -> data.Dataset:
    global INPUT_SIZE
    X_hat = teacher.predict(X, batch_size=512, verbose=0)
    y = loss(X, X_hat)
    y = quantize(np.log(y) * 32)
    return gen.get_generator(X.reshape((-1, INPUT_SIZE, 1)), y, 1024, True)


def train_model(
    model: Model,
    gen_train: tf.data.Dataset,
    gen_val: tf.data.Dataset,
    epoch: int = 1,
    steps: int = 1,
    callbacks=None,
    verbose: bool = False,
) -> None:
    model.fit(
        gen_train,
        steps_per_epoch=len(gen_train),
        initial_epoch=epoch,
        epochs=epoch + steps,
        validation_data=gen_val,
        callbacks=callbacks,
        verbose=verbose,
    )

def run_training(
     run_title: str, config: dict, eval_only: bool, epochs: int = 100, data_to_use: float = 0.1, Lambda = [0.0, 0.0], filters = [8, 12, 30], pooling = (2, 2),
     search: bool = False, verbose: bool = False
) -> None:
    global INPUT_SIZE

    t0=time.time()

    if search:
        run_title = f"{run_title}_epochs_{epochs}_data_{data_to_use}_search"
    else:
        run_title = f"{run_title}_epochs_{epochs}_data_{data_to_use}_lambda_{Lambda[0]}_{Lambda[1]}_filters_{filters[0]}_{filters[1]}_{filters[2]}"

    datasets_background = [i["path"] for i in config["background"] if i["use"]]
    datasets_background = [path for paths in datasets_background for path in paths]
    datasets_signal = [i["path"] for i in config["signal"] if i["use"]]
    datasets_signal = [path for paths in datasets_signal for path in paths]
    signal_names = ["Zero Bias", "SUEP", "HtoLongLived", "VBHFto2C", "TT", "SUSYGGBBH"]
    model_names_short = ["cic", "scn", "spr"]
    model_names_long = ["cicada", "section", "super"]


    gen = RegionETGenerator()
    X_train, X_val, X_test = gen.get_data_split(datasets_background, data_to_use)
    X_scn_train, X_scn_val, _ = gen.get_sectioned_data_split(datasets_background, data_to_use)
    X_spr_train, X_spr_val, _ = gen.get_super_data_split(datasets_background, data_to_use)

    X_test = [X_test]
    for i in range(len(datasets_signal)):
        X_train_tmp, X_val_tmp, X_test_tmp = gen.get_data_split([datasets_signal[i]], 1.0)
        X_tmp = np.concatenate((X_train_tmp, X_val_tmp, X_test_tmp), axis=0)
        X_test.append(X_tmp)


    bottleneck_sizes = np.array([np.arange(start=30, stop=93, step=3),
                                 np.arange(start=10, stop = 31, step=1),
                                 np.arange(start=10, stop = 31, step=1)])
    mse = np.zeros((len(model_names_short), len(signal_names), bottleneck_sizes.shape[1], 2))
    auc = np.zeros((len(model_names_short), len(signal_names)-1, bottleneck_sizes.shape[1], 2))

    for k in tqdm(range(bottleneck_sizes.shape[1])):

        if not os.path.exists(f"runs/{run_title}/plots/{bottleneck_sizes[0,k]}_{bottleneck_sizes[1,k]}"): os.makedirs(f"runs/{run_title}/plots/{bottleneck_sizes[0,k]}_{bottleneck_sizes[1,k]}")

        '''gen_train = gen.get_generator(X_train, X_train, 512, True)
        gen_val = gen.get_generator(X_val, X_val, 512)
        gen_scn_train = [gen.get_generator(np.reshape(X_scn_train[:,i], (-1,6,14,1)), np.reshape(X_scn_train[:,i], (-1,6,14,1)), 512) for i in range(3)]
        gen_scn_val = [gen.get_generator(np.reshape(X_scn_val[:,i], (-1,6,14,1)), np.reshape(X_scn_val[:,i], (-1,6,14,1)), 512) for i in range(3)]
        gen_spr_train = gen.get_generator(X_spr_train, X_spr_train, 512, True)
        gen_spr_val = gen.get_generator(X_spr_val, X_spr_val, 512)'''

        if not eval_only:
            teacher = TeacherAutoencoder((18, 14, 1), Lambda=[0.0, 0.0], filters=[20, 30, bottleneck_sizes[0,k]], pooling = (2, 2), search=False, compile=False, name=f"teacher_{bottleneck_sizes[0,k]}").get_model(hp=None)
            teachers_scn = [TeacherScnAutoencoder((6, 14, 1), Lambda=Lambda, filters=[8, 12, bottleneck_sizes[1,k]], pooling = pooling, search=False, compile=False, name=f"teacher_scn_{j+1}_{bottleneck_sizes[1,k]}").get_model(hp=None) for j in range(3)]
            teacher_spr = TeacherScnAutoencoder((6, 14, 1), Lambda=Lambda, filters=[8, 12, bottleneck_sizes[1,k]], pooling = pooling, search=False, compile=False, name=f"teacher_spr_{bottleneck_sizes[2,k]}").get_model(hp=None)

            teacher.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
            t_mc = ModelCheckpoint(f"runs/{run_title}/models/{teacher.name}", save_best_only=True)
            t_log = CSVLogger(f"runs/{run_title}/models/{teacher.name}/training.log", append=True)

            for i in range(3): teachers_scn[i].compile(optimizer=Adam(learning_rate=0.001), loss="mse")
            ts_scn_mc = [ModelCheckpoint(f"runs/{run_title}/models/teacher_scn_{i+1}_{bottleneck_sizes[1,k]}", save_best_only=True) for i in range(3)]
            ts_scn_log = [CSVLogger(f"runs/{run_title}/models/teacher_scn_{i+1}_{bottleneck_sizes[1,k]}/training.log", append=True) for i in range(3)]

            teacher_spr.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
            t_spr_mc = ModelCheckpoint(f"runs/{run_title}/models/teacher_spr_{bottleneck_sizes[2,k]}", save_best_only=True)
            t_spr_log = CSVLogger(f"runs/{run_title}/models/teacher_spr_{bottleneck_sizes[2,k]}/training.log", append=True)

            print(f"Training teachers on {X_train.shape[0]} events...")
            for epoch in tqdm(range(epochs)):
                train_model(teacher, gen_train, gen_val, epoch=epoch, callbacks=[t_mc, t_log], verbose=verbose)
                for i in range(3):
                    train_model(teachers_scn[i], gen_scn_train[i], gen_scn_val[i], epoch=epoch, callbacks=[ts_scn_mc[i], ts_scn_log[i]], verbose=verbose)
                train_model(teacher_spr, gen_spr_train, gen_spr_val, epoch=epoch, callbacks=[t_spr_mc, t_spr_log], verbose=verbose)

        draw = Draw(output_dir=f"runs/{run_title}/plots/{bottleneck_sizes[0,k]}_{bottleneck_sizes[1,k]}")

        teacher = keras.models.load_model(f"runs/{run_title}/models/teacher_{bottleneck_sizes[0,k]}")
        teachers_scn = [keras.models.load_model(f"runs/{run_title}/models/teacher_scn_{i+1}_{bottleneck_sizes[1,k]}") for i in range(3)]
        teacher_spr = keras.models.load_model(f"runs/{run_title}/models/teacher_spr_{bottleneck_sizes[2,k]}")

        print("Starting evaluation... plotting loss curves")
        # Plotting loss curves
        '''log = pd.read_csv(f"runs/{run_title}/models/teacher_{bottleneck_sizes[0,k]}/training.log")
        draw.plot_loss_history(log["loss"], log["val_loss"], "teacher_training_history", ylim=[1,3])
        log = []
        for i in range(3):
            log.append(pd.read_csv(f"runs/{run_title}/models/teacher_scn_{i+1}_{bottleneck_sizes[1,k]}/training.log"))
            draw.plot_loss_history(log[i]["loss"], log[i]["val_loss"], f"teacher_scn_{i+1}_training_history", ylim=[1,3])
        draw.plot_loss_history(np.sum([log[0]["loss"], log[1]["loss"], log[2]["loss"]], axis = 0), np.sum([log[0]["val_loss"], log[1]["val_loss"], log[2]["val_loss"]], axis = 0), "teacher_scn_training_history_sum", ylim=[1,3])
        draw.plot_multiple_loss_history([[log[0]["loss"], log[0]["val_loss"], f"teacher_scn_0"],
                                         [log[1]["loss"], log[1]["val_loss"], f"teacher_scn_1"],
                                         [log[2]["loss"], log[2]["val_loss"], f"teacher_scn_2"]],
                                         "teacher_scn_training_history_overlay", ylim=[1,3])
        log = pd.read_csv(f"runs/{run_title}/models/teacher_spr_{bottleneck_sizes[2,k]}/training.log")
        draw.plot_loss_history(log["loss"], log["val_loss"], "teacher_spr_training_history", ylim=[1,3])'''

        print("Finished plotting loss curves... plotting reconstruction examples")
        # Plotting reconstruction examples
        X_test_len = np.zeros(6)
        for i in range(X_test_len.shape[0]):
            X_test_len[i] = X_test[i].shape[0]
        X_all_test = np.concatenate((X_test), axis=0)
        y_pred_cic = teacher.predict(X_all_test, batch_size=512, verbose=verbose)
        y_loss_cic = loss(X_all_test, y_pred_cic)
        X_scn_reshape = np.zeros((X_all_test.shape[0]*3,6,14,1))
        for i in range(X_all_test.shape[0]):
            for j in range(3):
                X_scn_reshape[i*3+j] = X_all_test[i, j*6:j*6+6]
        y_scn_reshape = teachers_scn[j].predict(X_scn_reshape, batch_size=512, verbose=verbose)
        y_spr_reshape = teacher_spr.predict(X_scn_reshape, batch_size=512, verbose=verbose)
        y_pred_scn = np.zeros((X_all_test.shape[0], 18, 14, 1))
        y_pred_spr = np.zeros((X_all_test.shape[0], 18, 14, 1))
        for i in range(X_all_test.shape[0]):
            y_scn_tmp = np.zeros((18, 14, 1))
            y_spr_tmp = np.zeros((18, 14, 1))
            for j in range(3):
                y_scn_tmp[j*6:j*6+6] = y_scn_reshape[i*3+j]
                y_spr_tmp[j*6:j*6+6] = y_spr_reshape[i*3+j]
            y_pred_scn[i] = y_scn_tmp
            y_pred_spr[i] = y_spr_tmp
        y_loss_scn = loss(X_all_test, y_pred_scn)
        y_loss_spr = loss(X_all_test, y_pred_spr)
        y_pred, y_loss = [[], [], []], [[], [], []]
        tmp = 0
        for i in range(X_test_len.shape[0]):
            y_pred[0].append(y_pred_cic[int(tmp):int(tmp+X_test_len[i])])
            y_pred[1].append(y_pred_scn[int(tmp):int(tmp+X_test_len[i])])
            y_pred[2].append(y_pred_spr[int(tmp):int(tmp+X_test_len[i])])
            y_loss[0].append(y_loss_cic[int(tmp):int(tmp+X_test_len[i])])
            y_loss[1].append(y_loss_scn[int(tmp):int(tmp+X_test_len[i])])
            y_loss[2].append(y_loss_spr[int(tmp):int(tmp+X_test_len[i])])
            tmp += int(X_test_len[i])
        for i in range(len(model_names_short)):
            for j in range(len(signal_names)):
                mse[i,j,k,0] = np.mean(y_loss[i][j])
                mse[i,j,k,1] = np.std(y_loss[i][j])
        '''for i in range(10):
            for j in range(3):
                draw.plot_reconstruction_results(np.array([X_test[0][i]]), np.array([y_pred[j][0][i]]), loss=y_loss[j][0][i], name=f"comparison_background_{model_names_short[j]}_{i}")'''

        print("Finished plotting reconstruction examples... plotting roc curves")
        # Declaring lists to be used in roc curve
        y_true, inputs, y_pred_roc = [], [], [[], [], []]
        y_pred_roc_sig = [[], [], [], [], [], []]
        for i in range(1, len(signal_names)):
            inputs.append(np.concatenate((X_test[i], X_test[0])))
            y_true.append(np.concatenate((np.ones(int(X_test_len[i])), np.zeros(int(X_test_len[0])))))
            for j in range(len(model_names_short)):
                y_pred_roc[j].append(np.concatenate((y_loss[j][i], y_loss[j][0])))
                y_pred_roc_sig[i].append(np.concatenate((y_loss[j][i], y_loss[j][0])))
        for i in range(len(model_names_short)):
            draw.plot_roc_curve(y_true, y_pred_roc[i], signal_names, inputs, f"teacher_{model_names_short[i]}")
        for i in range(len(signal_names)-1):
            auc[:,i,k,:] = draw.plot_roc_curve([y_true[i],y_true[i],y_true[i]], y_pred_roc_sig[i+1], model_names_long, [inputs[i],inputs[i],inputs[i]], f"signal_{signal_names[i+1]}")
        print("Finished plotting roc curves... plotting anomaly score distribution")
        # Declaring dicts to be used in anomaly score distribution
        '''results, results_sig = [dict(), dict(), dict()], [dict(), dict(), dict(), dict(), dict(), dict()]
        list_results, list_results_sig = [], []
        for i in range(3):
            for j in range(len(signal_names)):
                results[i][f'{signal_names[j]}'] = y_loss[i][j]
            list_results.append(list(results[i].values()))
            draw.plot_anomaly_score_distribution(list_results[i], [*results[i]], f"anomaly_score_teacher_{model_names_short[i]}", xlim=[0,256])
        for i in range(len(signal_names)):
            for j in range(3):
                results_sig[i][f'{model_names_long[j]}'] = y_loss[j][i]
            list_results_sig.append(list(results_sig[i].values()))
            draw.plot_anomaly_score_distribution(list_results_sig[i], [*results_sig[i]], f"anomaly_score_signal_{signal_names[i]}", xlim=[0,256])
        print("Finished plotting anomaly score distribution... plotting scatter plots and finding score correlations")
        # Anomaly score comparison statistics
        corr, scatter_fit, emd = [[], []], [[], []], [[], []]
        for i in range(len(signal_names)):
            for j in range(2):
                corr[j].append(np.corrcoef(y_loss[0][i], y_loss[j+1][i]))
                scatter_fit[j].append(draw.plot_scatter_score_comparison(np.sqrt(y_loss[0][i]), np.sqrt(y_loss[j+1][i]), f"{model_names_short[0]}", f"{model_names_short[j+1]}", f"{model_names_short[0]}_{model_names_short[j+1]}_{signal_names[i]}", limits="equalsignal"))
                emd[j].append(wasserstein_distance(np.sqrt(y_loss[0][i]), np.sqrt(y_loss[j+1][i])))
        score_dist = draw.plot_score_comparison_distributions(y_loss, f"Difference", f"Frequency", name="tea", limits="equalsignal")

        f = open(f"runs/{run_title}/plots/{bottleneck_sizes[0,k]}_{bottleneck_sizes[1,k]}/summary.txt", "w")
        f.write(f"Trained on {X_train.shape[0]} events\n")
        for i in range(len(signal_names)):
            for j in range(2):
                f.write(f"{model_names_short[j+1]}_{signal_names[i]}:\n")
                f.write(f"corr: {corr[j][i]}\n")
                f.write(f"M: {scatter_fit[j][i][0]}\n")
                f.write(f"s: {score_dist[j][i][0]}\n")
                f.write(f"mu: {score_dist[j][i][1]}\n")
                f.write(f"emd: {emd[j][i]}\n")
        f.close()'''

    draw = Draw(output_dir=f"runs/{run_title}/plots")
    draw.plot_mse(mse, bottleneck_sizes)
    draw.plot_auc(auc, bottleneck_sizes)
    f = open(f"runs/{run_title}/correlation.txt", "w")
    t1=time.time()
    f.write(f"Total time: {t1-t0}\n")
    f.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description="""CICADA sectioned training scripts""")
    parser.add_argument(
        "run_title",
        type=str,
        help="Title of run",
    )
    parser.add_argument(
        "-epochs",
        "-epochs",
        type=int,
        help="Number of training epochs",
        default=100,
    )
    parser.add_argument(
        "-data",
        "-data",
        type=float,
        help="Fraction of data to include",
        default=0.1,
    )
    parser.add_argument(
        "-Lambda",
        "-Lambda",
        type=float,
        nargs="+",
        help="Lambdas to use for L2 regularization, [0] for encoder layers, [1] for decoder layers. Accepts floats.",
        default=[0.0, 0.0],
    )
    parser.add_argument(
        "-filters",
        "-filters",
        type=int,
        nargs="+",
        help="nFilters to use for Conv2D layers, [0] for 1st encoder layer and 2nd decoder layer, [1] for 2nd encoder layer and 1st decoder layer, [2] for bottleneck layer. Accepts ints.",
        default=[8, 12, 20],
    )
    parser.add_argument(
        "-pool",
        "-pool",
        type=int,
        nargs="+",
        help="Pooling dimension. [0] for phi direction, [1] for eta direction. Accepts ints.",
        default=[2, 2],
    )
    parser.add_argument(
        "-search",
        action="store_true",
        help="Conduct search over regularization and filter hyperparameters?",
        default=False,
    )
    parser.add_argument(
        "-config",
        "-config",
        action=IsValidFile,
        type=Path,
        default="misc/config.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "-evaluate-only",
        action="store_true",
        help="Skip training",
        default=False,
    )
    parser.add_argument(
        "-verbose",
        "--verbose",
        action="store_true",
        help="Output verbosity",
        default=False,
    )
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    return args, config


def main(args_in=None) -> None:
    args, config = parse_arguments()
    run_training(run_title=args.run_title, config=config, eval_only=args.evaluate_only, epochs=args.epochs, data_to_use=args.data, Lambda=args.Lambda, filters=args.filters, pooling=tuple(args.pool), search=args.search, verbose=args.verbose)


if __name__ == "__main__":
    main()
