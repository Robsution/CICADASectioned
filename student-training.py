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
from models import TeacherAutoencoder, TeacherScnAutoencoder, CicadaV1, CicadaV1scn, CicadaV2, CicadaV2scn
from pathlib import Path
from tensorflow import data
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from typing import List
from utils import IsValidFile
from scipy.stats import wasserstein_distance

from qkeras import *
from tqdm import tqdm


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
    teacher: Model, gen: RegionETGenerator, X: npt.NDArray, input_size: int
) -> data.Dataset:
    X_hat = teacher.predict(X, batch_size=512, verbose=0)
    y = loss(X, X_hat)
    y = quantize(np.log(y) * 32)
    return gen.get_generator(X.reshape((-1, input_size, 1)), y, 1024, True)


def train_model(
    model: Model,
    gen_train: data.Dataset,
    gen_val: data.Dataset,
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
     run_title: str, config: dict, train_teachers: bool, train_students: bool, eval: bool, epochs: int = 100, data_to_use: float = 0.1,
     Lambda = [0.0, 0.0], filters = [8, 12, 20], pooling = [2, 2],
     dense1 = [8], dropout1 = [0.1], dense2 = [8], dropout2 = [0.1, 0.1], filters2 = [2],
     search: bool = False, verbose: bool = False
) -> None:

    t0=time.time()

    if search: run_title = f"{run_title}_epochs_{epochs}_data_{data_to_use}_search"
    else: run_title = f"{run_title}_epochs_{epochs}_data_{data_to_use}_lambda_{Lambda[0]}_{Lambda[1]}_filters_{filters[0]}_{filters[1]}_{filters[2]}"
    draw = Draw(output_dir=f"runs/{run_title}/plots")

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
        X_tmp = np.concatenate(gen.get_data_split([datasets_signal[i]], 1.0), axis=0)
        X_test.append(X_tmp)

    gen_train = gen.get_generator(X_train, X_train, 512, True)
    gen_val = gen.get_generator(X_val, X_val, 512)
    gen_scn_train = [gen.get_generator(np.reshape(X_scn_train[:,i], (-1,6,14,1)), np.reshape(X_scn_train[:,i], (-1,6,14,1)), 512) for i in range(3)]
    gen_scn_val = [gen.get_generator(np.reshape(X_scn_val[:,i], (-1,6,14,1)), np.reshape(X_scn_val[:,i], (-1,6,14,1)), 512) for i in range(3)]
    gen_spr_train = gen.get_generator(X_spr_train, X_spr_train, 512, True)
    gen_spr_val = gen.get_generator(X_spr_val, X_spr_val, 512)

    outlier_train = gen.get_data(config["exposure"]["training"])
    outlier_val = gen.get_data(config["exposure"]["validation"])
    X_train_student = np.concatenate([X_train, outlier_train])
    X_val_student = np.concatenate([X_val, outlier_train])
    outlier_train_scn = gen.get_sectioned_data(config["exposure"]["training"])
    outlier_val_scn = gen.get_sectioned_data(config["exposure"]["validation"])
    X_train_student_scn, X_val_student_scn = [[], [], []], [[], [], []]
    for i in range(3):
        X_train_student_scn[i] = np.concatenate([X_scn_train[i], outlier_train_scn[i]])
        X_val_student_scn[i] = np.concatenate([X_scn_val[i], outlier_train_scn[i]])
    outlier_train_spr = gen.get_super_data(config["exposure"]["training"])
    outlier_val_spr = gen.get_super_data(config["exposure"]["validation"])
    X_train_student_spr = np.concatenate([X_spr_train, outlier_train_spr])
    X_val_student_spr = np.concatenate([X_spr_val, outlier_train_spr])

    X_train_student = [X_train_student, X_train_student_scn[0],  X_train_student_scn[1],  X_train_student_scn[2],  X_train_student_spr]
    X_val_student = [X_val_student, X_val_student_scn[0],  X_val_student_scn[1],  X_val_student_scn[2],  X_val_student_spr]

    t1=time.time()

    if search:
        paths = ["teacher", "teacher_scn_1", "teacher_scn_2", "teacher_scn_3", "teacher_spr"]
        for path in paths:
            if not os.path.exists(f"runs/{run_title}/models/search/" + path):
                os.makedirs(f"runs/{run_title}/models/search/" + path)

        teacher_tuner = kt.Hyperband(
            hypermodel=TeacherAutoencoder((18, 14, 1), search=search, compile=True, name="teacher_tuner").get_model,
            objective=kt.Objective("val_mean_squared_error", direction="min"),
            max_epochs=epochs,
            seed=42,
            overwrite=True,
            directory=f"runs/{run_title}/models/search",
            project_name="teacher")
        teacher_tuner_scn = []
        for i in range(3):
            teacher_tuner_scn.append(kt.Hyperband(
                hypermodel=TeacherScnAutoencoder((6, 14, 1), search=search, compile=True, name=f"teacher_tuner_scn_{i+1}").get_model,
                objective=kt.Objective("val_mean_squared_error", direction="min"),
                max_epochs=epochs,
                seed=42,
                overwrite=True,
                directory=f"runs/{run_title}/models/search",
                project_name=f"teacher_scn_{i+1}"))
        teacher_tuner_spr = kt.Hyperband(
            hypermodel=TeacherScnAutoencoder((6, 14, 1), search=search, compile=True, name=f"teacher_tuner_spr").get_model,
            objective=kt.Objective("val_mean_squared_error", direction="min"),
            max_epochs=epochs,
            seed=42,
            overwrite=True,
            directory=f"runs/{run_title}/models/search",
            project_name=f"teacher_spr")
        teacher_tuner.search(x=X_train, y=X_train, validation_data=(X_val, X_val))
        for i in range(3): teacher_tuner_scn[i].search(x=X_scn_train[i], y=X_scn_train[i], validation_data=(X_scn_val[i], X_scn_val[i]))
        teacher_tuner_spr.search(x=X_spr_train, y=X_spr_train, validation_data=(X_spr_val, X_spr_val))
        teacher_hp = teacher_tuner.get_best_hyperparameters()[0]
        teacher_hp_scn = []
        for i in range(3): teacher_hp_scn.append(teacher_tuner_scn[i].get_best_hyperparameters()[0])
        teacher_hp_spr = teacher_tuner_spr.get_best_hyperparameters()[0]
        f = open(f"runs/{run_title}/search.txt", "w")
        teacher_tuner.results_summary()
        for i in range(3): teacher_tuner_scn[i].results_summary()
        teacher_tuner_spr.results_summary()
        f.close()

        teacher = TeacherAutoencoder((18, 14, 1), search=search, compile=False, name="teacher").get_model(hp=teacher_hp)
        teachers_scn = [TeacherScnAutoencoder((6, 14, 1), search=search, compile=False, name=f"teacher_scn_{i+1}").get_model(hp=teacher_hp_scn[i]) for i in range(3)]
        teacher_spr = TeacherScnAutoencoder((6, 14, 1), search=search, compile=True, name="teacher_spr").get_model(hp=teacher_hp_spr)
    else:
        teacher = TeacherAutoencoder((18, 14, 1), Lambda=[0.0, 0.0], filters=[20, 30, 80], pooling = (2, 2), search=False, compile=False, name="teacher").get_model(hp=None)
        teachers_scn = [TeacherScnAutoencoder((6, 14, 1), Lambda=Lambda, filters=filters, pooling = pooling, search=False, compile=False, name=f"teacher_scn_{i+1}").get_model(hp=None) for i in range(3)]
        teacher_spr = TeacherScnAutoencoder((6, 14, 1), Lambda=Lambda, filters=filters, pooling = pooling, search=False, compile=False, name="teacher_spr").get_model(hp=None)

    if train_teachers:
        teacher.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        t_mc = ModelCheckpoint(f"runs/{run_title}/models/{teacher.name}", save_best_only=True)
        t_log = CSVLogger(f"runs/{run_title}/models/{teacher.name}/training.log", append=True)

        for i in range(3): teachers_scn[i].compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        t_scn_mc = [ModelCheckpoint(f"runs/{run_title}/models/{teachers_scn[i].name}", save_best_only=True) for i in range(3)]
        t_scn_log = [CSVLogger(f"runs/{run_title}/models/{teachers_scn[i].name}/training.log", append=True) for i in range(3)]

        teacher_spr.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        t_spr_mc = ModelCheckpoint(f"runs/{run_title}/models/{teacher_spr.name}", save_best_only=True)
        t_spr_log = CSVLogger(f"runs/{run_title}/models/{teacher_spr.name}/training.log", append=True)

        print(f"Training teachers on {X_train.shape[0]} events...")
        for epoch in tqdm(range(epochs)):
            train_model(teacher, gen_train, gen_val, epoch=epoch, callbacks=[t_mc, t_log], verbose=verbose)
            for i in range(3):
                train_model(teachers_scn[i], gen_scn_train[i], gen_scn_val[i], epoch=epoch, callbacks=[t_scn_mc[i], t_scn_log[i]], verbose=verbose)
            train_model(teacher_spr, gen_spr_train, gen_spr_val, epoch=epoch, callbacks=[t_spr_mc, t_spr_log], verbose=verbose)

    tea = [load_model(f"runs/{run_title}/models/teacher"),
           load_model(f"runs/{run_title}/models/teacher_scn_1"),
           load_model(f"runs/{run_title}/models/teacher_scn_2"),
           load_model(f"runs/{run_title}/models/teacher_scn_3"),
           load_model(f"runs/{run_title}/models/teacher_spr")]

    t2=time.time()

    if train_students:
        cv1_cic = CicadaV1((252,), name="cv1_cic").get_model()
        cv1_cic.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
        cv1_cic_mc = ModelCheckpoint(f"runs/{run_title}/models/{cv1_cic.name}", save_best_only=True)
        cv1_cic_log = CSVLogger(f"runs/{run_title}/models/{cv1_cic.name}/training.log", append=True)

        cv1_scn = [CicadaV1scn((84,), dense=dense1, dropout=dropout1, search=False, compile=False, name=f"cv1_scn_{i+1}").get_model(hp=None) for i in range(3)]
        for i in range(3): cv1_scn[i].compile(optimizer=Adam(learning_rate=0.001), loss="mae")
        cv1_scn_mc = [ModelCheckpoint(f"runs/{run_title}/models/{cv1_scn[i].name}", save_best_only=True) for i in range(3)]
        cv1_scn_log = [CSVLogger(f"runs/{run_title}/models/{cv1_scn[i].name}/training.log", append=True) for i in range(3)]

        cv1_spr = CicadaV1scn((84,), dense=dense1, dropout=dropout1, search=False, compile=False, name="cv1_spr").get_model(hp=None)
        cv1_spr.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
        cv1_spr_mc = ModelCheckpoint(f"runs/{run_title}/models/{cv1_spr.name}", save_best_only=True)
        cv1_spr_log = CSVLogger(f"runs/{run_title}/models/{cv1_spr.name}/training.log", append=True)

        cv2_cic = CicadaV2((252,), name="cv2_cic").get_model()
        cv2_cic.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
        cv2_cic_mc = ModelCheckpoint(f"runs/{run_title}/models/{cv2_cic.name}", save_best_only=True)
        cv2_cic_log = CSVLogger(f"runs/{run_title}/models/{cv2_cic.name}/training.log", append=True)

        cv2_scn = [CicadaV2scn((84,), dense=dense2, dropout=dropout2, filters=filters2, search=False, compile=False, name=f"cv2_scn_{i+1}").get_model(hp=None) for i in range(3)]
        for i in range(3): cv2_scn[i].compile(optimizer=Adam(learning_rate=0.001), loss="mae")
        cv2_scn_mc = [ModelCheckpoint(f"runs/{run_title}/models/{cv2_scn[i].name}", save_best_only=True) for i in range(3)]
        cv2_scn_log = [CSVLogger(f"runs/{run_title}/models/{cv2_scn[i].name}/training.log", append=True) for i in range(3)]

        cv2_spr = CicadaV2scn((84,), dense=dense2, dropout=dropout2, filters=filters2, search=False, compile=False, name="cv2_spr").get_model(hp=None)
        cv2_spr.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
        cv2_spr_mc = ModelCheckpoint(f"runs/{run_title}/models/{cv2_spr.name}", save_best_only=True)
        cv2_spr_log = CSVLogger(f"runs/{run_title}/models/{cv2_spr.name}/training.log", append=True)

        cv1 = [cv1_cic, cv1_scn[0], cv1_scn[1], cv1_scn[2], cv1_spr]
        cv1_mc = [cv1_cic_mc, cv1_scn_mc[0], cv1_scn_mc[1], cv1_scn_mc[2], cv1_spr_mc]
        cv1_log = [cv1_cic_log, cv1_scn_log[0], cv1_scn_log[1], cv1_scn_log[2], cv1_spr_log]

        cv2 = [cv2_cic, cv2_scn[0], cv2_scn[1], cv2_scn[2], cv2_spr]
        cv2_mc = [cv2_cic_mc, cv2_scn_mc[0], cv2_scn_mc[1], cv2_scn_mc[2], cv2_spr_mc]
        cv2_log = [cv2_cic_log, cv2_scn_log[0], cv2_scn_log[1], cv2_scn_log[2], cv2_spr_log]

        print(f"Training students on {X_train.shape[0]} events...")
        input_sizes = [252, 84, 84, 84, 84]
        for epoch in tqdm(range(epochs)):
            s_gen_train, s_gen_val = [], []
            for i in range(len(tea)):
                s_gen_train.append(get_student_targets(tea[i], gen, X_train_student[i], input_sizes[i]))
                s_gen_val.append(get_student_targets(tea[i], gen, X_val_student[i], input_sizes[i]))
            for i in range(len(input_sizes)):
                train_model(cv1[i], s_gen_train[i], s_gen_val[i], epoch=10*epoch, steps=10, callbacks=[cv1_mc[i], cv1_log[i]], verbose=verbose)
                train_model(cv2[i], s_gen_train[i], s_gen_val[i], epoch=10*epoch, steps=10, callbacks=[cv2_mc[i], cv2_log[i]], verbose=verbose)

    cv1 = [load_model(f"runs/{run_title}/models/cv1_cic"),
           load_model(f"runs/{run_title}/models/cv1_scn_1"),
           load_model(f"runs/{run_title}/models/cv1_scn_2"),
           load_model(f"runs/{run_title}/models/cv1_scn_3"),
           load_model(f"runs/{run_title}/models/cv1_spr")]
    cv2 = [load_model(f"runs/{run_title}/models/cv2_cic"),
           load_model(f"runs/{run_title}/models/cv2_scn_1"),
           load_model(f"runs/{run_title}/models/cv2_scn_2"),
           load_model(f"runs/{run_title}/models/cv2_scn_3"),
           load_model(f"runs/{run_title}/models/cv2_spr")]

    t3=time.time()

    if eval:
        if not os.path.exists(f"runs/{run_title}/plots"): os.makedirs(f"runs/{run_title}/plots")

        f = open(f"runs/{run_title}/model_summary.txt", "w")
        for models in [tea, cv1, cv2]:
            for model in models:
                 model.summary(print_fn=f.write)
                 f.write("\n\n")
        f.close()

        print("Starting evaluation... plotting loss curves")
        # Plotting loss curves (teacher)
        log = pd.read_csv(f"runs/{run_title}/models/teacher/training.log")
        draw.plot_loss_history(log["loss"], log["val_loss"], "teacher_training_history", ylim=[1,5])
        log = []
        for i in range(3):
            log.append(pd.read_csv(f"runs/{run_title}/models/teacher_scn_{i+1}/training.log"))
            draw.plot_loss_history(log[i]["loss"],
                                   log[i]["val_loss"],
                                   f"teacher_scn_{i+1}_training_history", ylim=[1,5])
        draw.plot_loss_history(np.sum([log[0]["loss"], log[1]["loss"], log[2]["loss"]], axis = 0),
                               np.sum([log[0]["val_loss"], log[1]["val_loss"], log[2]["val_loss"]], axis = 0),
                               "teacher_scn_training_history_sum", ylim=[1,5])
        draw.plot_multiple_loss_history([[log[0]["loss"], log[0]["val_loss"], f"teacher_scn_0"],
                                         [log[1]["loss"], log[1]["val_loss"], f"teacher_scn_1"],
                                         [log[2]["loss"], log[2]["val_loss"], f"teacher_scn_2"]],
                                         "teacher_scn_training_history_overlay", ylim=[1,5])
        log = pd.read_csv(f"runs/{run_title}/models/teacher_spr/training.log")
        draw.plot_loss_history(log["loss"], log["val_loss"], "teacher_spr_training_history", ylim=[1,5])
        # Plotting loss curves (student)
        cv1_log = [pd.read_csv(f"runs/{run_title}/models/{cv1[i].name}/training.log") for i in range(len(cv1))]
        cv2_log = [pd.read_csv(f"runs/{run_title}/models/{cv2[i].name}/training.log") for i in range(len(cv2))]
        for cv_log, cv, i in zip([cv1_log, cv2_log], [cv1, cv2], range(2)):
            for j in range(len(cv_log)):
                draw.plot_loss_history(cv_log[j]["loss"],
                                       cv_log[j]["val_loss"],
                                       f"{cv[j].name}_training_history_zoom_out", ylim=[0, 20])
                draw.plot_loss_history(cv_log[j]["loss"],
                                       cv_log[j]["val_loss"],
                                       f"{cv[j].name}_training_history", ylim=[1, 10])
            draw.plot_multiple_loss_history([[cv_log[1]["loss"], cv_log[1]["val_loss"], f"cv{i+1}_scn_1"],
                                             [cv_log[2]["loss"], cv_log[2]["val_loss"], f"cv{i+1}_scn_2"],
                                             [cv_log[3]["loss"], cv_log[3]["val_loss"], f"cv{i+1}_scn_3"]],
                                             f"cv{i+1}_scn_training_history_overlay_zoom_out", ylim=[0, 20])
            draw.plot_multiple_loss_history([[cv_log[1]["loss"], cv_log[1]["val_loss"], f"cv{i+1}_scn_1"],
                                             [cv_log[2]["loss"], cv_log[2]["val_loss"], f"cv{i+1}_scn_2"],
                                             [cv_log[3]["loss"], cv_log[3]["val_loss"], f"cv{i+1}_scn_3"]],
                                             f"cv{i+1}_scn_training_history_overlay", ylim=[1, 10])

        print("Finished plotting loss curves... calculating scores")
        # Calculating teacher scores
        X_test_len = np.zeros(6)
        X_test_len_cum = np.zeros(7)
        for i in range(X_test_len.shape[0]):
            X_test_len[i] = X_test[i].shape[0]
        for i in range(X_test_len.shape[0]):
            X_test_len_cum[i+1] = X_test_len[i] + X_test_len_cum[i]
        X_test_len = X_test_len.astype(int)
        X_test_len_cum = X_test_len_cum.astype(int)
        X_all_test = np.concatenate((X_test), axis=0)
        y_tea_pred_cic = tea[0].predict(X_all_test, batch_size=512, verbose=verbose)
        y_tea_loss_cic = loss(X_all_test, y_tea_pred_cic)
        X_scn_reshape = np.zeros((3,X_all_test.shape[0],6,14,1))
        y_tea_scn_reshape = np.zeros((3,X_all_test.shape[0],6,14,1))
        y_tea_spr_reshape = np.zeros((3,X_all_test.shape[0],6,14,1))
        for i in range(3):
            X_scn_reshape[i] = X_all_test[:, i*6:i*6+6]
            y_tea_scn_reshape[i] = tea[i+1].predict(np.array(X_scn_reshape[i]), batch_size=512, verbose=verbose)
            y_tea_spr_reshape[i] = tea[4].predict(np.array(X_scn_reshape[i]), batch_size=512, verbose=verbose)
        y_tea_pred_scn = np.zeros((X_all_test.shape[0], 18, 14, 1))
        y_tea_pred_spr = np.zeros((X_all_test.shape[0], 18, 14, 1))
        for i in range(X_all_test.shape[0]):
            y_tea_scn_tmp = np.zeros((18, 14, 1))
            y_tea_spr_tmp = np.zeros((18, 14, 1))
            for j in range(3):
                y_tea_scn_tmp[j*6:j*6+6] = y_tea_scn_reshape[j][i]
                y_tea_spr_tmp[j*6:j*6+6] = y_tea_spr_reshape[j][i]
            y_tea_pred_scn[i] = y_tea_scn_tmp
            y_tea_pred_spr[i] = y_tea_spr_tmp
        y_tea_loss_scn = loss(X_all_test, y_tea_pred_scn)
        y_tea_loss_spr = loss(X_all_test, y_tea_pred_spr)
        y_tea_pred, y_tea_loss = [[], [], []], [[], [], []]
        for i in range(len(signal_names)):
            y_tea_pred[0].append(y_tea_pred_cic[X_test_len_cum[i]:X_test_len_cum[i+1]])
            y_tea_pred[1].append(y_tea_pred_scn[X_test_len_cum[i]:X_test_len_cum[i+1]])
            y_tea_pred[2].append(y_tea_pred_spr[X_test_len_cum[i]:X_test_len_cum[i+1]])
            y_tea_loss[0].append(y_tea_loss_cic[X_test_len_cum[i]:X_test_len_cum[i+1]])
            y_tea_loss[1].append(y_tea_loss_scn[X_test_len_cum[i]:X_test_len_cum[i+1]])
            y_tea_loss[2].append(y_tea_loss_spr[X_test_len_cum[i]:X_test_len_cum[i+1]])
        # Calculating student scores
        y_cv1_loss, y_cv2_loss = [[], [], []], [[], [], []]
        for cv, y_cv_loss in zip([cv1, cv2], [y_cv1_loss, y_cv2_loss]):
            y_cv_loss_cic = cv[0].predict(X_all_test.reshape(-1,252,1), batch_size=512, verbose=verbose)
            y_cv_loss_scn = np.zeros((3,X_all_test.shape[0],1))
            y_cv_loss_spr = np.zeros((3,X_all_test.shape[0],1))
            for i in range(3):
                y_cv_loss_scn[i] = cv[i+1].predict(np.array(X_scn_reshape[i].reshape(-1, 84, 1)), batch_size=512, verbose=verbose)
                y_cv_loss_spr[i] = cv[4].predict(np.array(X_scn_reshape[i].reshape(-1, 84, 1)), batch_size=512, verbose=verbose)
            y_cv_loss_scn = np.sum(y_cv_loss_scn, axis=0) / 3.
            y_cv_loss_spr = np.sum(y_cv_loss_spr, axis=0) / 3.
            for i in range(len(signal_names)):
                y_cv_loss[0].append(y_cv_loss_cic[X_test_len_cum[i]:X_test_len_cum[i+1]])
                y_cv_loss[1].append(y_cv_loss_scn[X_test_len_cum[i]:X_test_len_cum[i+1]])
                y_cv_loss[2].append(y_cv_loss_spr[X_test_len_cum[i]:X_test_len_cum[i+1]])
        '''y_cv1_loss_cic = cv1[0].predict(X_all_test.reshape(-1,252,1), batch_size=512, verbose=verbose)
        y_cv2_loss_cic = cv2[0].predict(X_all_test.reshape(-1,252,1), batch_size=512, verbose=verbose)
        y_cv1_loss_scn = np.zeros((3,X_all_test.shape[0],1))
        y_cv1_loss_spr = np.zeros((3,X_all_test.shape[0],1))
        y_cv2_loss_scn = np.zeros((3,X_all_test.shape[0],1))
        y_cv2_loss_spr = np.zeros((3,X_all_test.shape[0],1))
        for i in range(3):
            y_cv1_loss_scn[i] = cv1[i+1].predict(np.array(X_scn_reshape[i].reshape(-1, 84, 1)), batch_size=512, verbose=verbose)
            y_cv2_loss_spr[i] = cv2[i+1].predict(np.array(X_scn_reshape[i].reshape(-1, 84, 1)), batch_size=512, verbose=verbose)
            y_cv1_loss_scn[i] = cv1[4].predict(np.array(X_scn_reshape[i].reshape(-1, 84, 1)), batch_size=512, verbose=verbose)
            y_cv2_loss_spr[i] = cv2[4].predict(np.array(X_scn_reshape[i].reshape(-1, 84, 1)), batch_size=512, verbose=verbose)
        y_cv1_loss_scn = np.sum(y_cv1_loss_scn, axis=0)
        y_cv1_loss_spr = np.sum(y_cv1_loss_spr, axis=0)
        y_cv2_loss_scn = np.sum(y_cv2_loss_scn, axis=0)
        y_cv2_loss_spr = np.sum(y_cv2_loss_spr, axis=0)
        y_cv1_loss, y_cv2_loss = [[], [], []], [[], [], []]
        for i in range(len(signal_names)):
            y_cv1_loss[0].append(y_cv1_loss_cic[X_test_len_cum[i]:X_test_len_cum[i+1]])
            y_cv1_loss[1].append(y_cv1_loss_scn[X_test_len_cum[i]:X_test_len_cum[i+1]])
            y_cv1_loss[2].append(y_cv1_loss_spr[X_test_len_cum[i]:X_test_len_cum[i+1]])
            y_cv2_loss[0].append(y_cv2_loss_cic[X_test_len_cum[i]:X_test_len_cum[i+1]])
            y_cv2_loss[1].append(y_cv2_loss_scn[X_test_len_cum[i]:X_test_len_cum[i+1]])
            y_cv2_loss[2].append(y_cv2_loss_spr[X_test_len_cum[i]:X_test_len_cum[i+1]])'''
            # y_loss has shape (3, 6, nEvents, 1)
        for i in range(10):
            for j in range(3):
               draw.plot_reconstruction_results(np.array([X_test[0][i]]), np.array([y_tea_pred[j][0][i]]), loss=y_tea_loss[j][0][i], name=f"comparison_background_{model_names_short[j]}_{i}")

        print("Finished calculating scores... plotting roc curves")
        # Declaring lists to be used in roc curve
        for y_loss, name in zip([y_tea_loss, y_cv1_loss, y_cv2_loss], ["tea", "cv1", "cv2"]):
            y_true, inputs, y_loss_roc, y_loss_roc_sig = [], [], [[], [], []], [[], [], [], [], [], []]
            for i in range(1, len(signal_names)):
                inputs.append(np.concatenate((X_test[i], X_test[0])))
                y_true.append(np.concatenate((np.ones(int(X_test_len[i])), np.zeros(int(X_test_len[0])))))
                for j in range(len(model_names_short)):
                    y_loss_roc[j].append(np.concatenate((y_loss[j][i], y_loss[j][0])))
                    y_loss_roc_sig[i].append(np.concatenate((y_loss[j][i], y_loss[j][0])))
            for i in range(len(model_names_short)):
                draw.plot_roc_curve(y_true, y_loss_roc[i], signal_names, inputs, f"{name}_{model_names_short[i]}")
            for i in range(len(signal_names)-1):
                draw.plot_roc_curve([y_true[i],y_true[i],y_true[i]], y_loss_roc_sig[i+1], model_names_long, [inputs[i],inputs[i],inputs[i]], f"{name}_signal_{signal_names[i+1]}")
        '''y_true, inputs, y_tea_loss_roc, y_tea_loss_roc_sig = [], [], [[], [], []], [[], [], [], [], [], []]
        y_cv1_loss_roc, y_cv1_loss_roc_sig, y_cv2_loss_roc, y_cv2_loss_roc_sig = [[], [], []], [[], [], [], [], [], []], [[], [], []], [[], [], [], [], [], []]
        for i in range(1, len(signal_names)):
            inputs.append(np.concatenate((X_test[i], X_test[0])))
            y_true.append(np.concatenate((np.ones(int(X_test_len[i])), np.zeros(int(X_test_len[0])))))
            for j in range(len(model_names_short)):
                y_tea_loss_roc[j].append(np.concatenate((y_tea_loss[j][i], y_tea_loss[j][0])))
                y_tea_loss_roc_sig[i].append(np.concatenate((y_tea_loss[j][i], y_tea_loss[j][0])))
                y_cv1_loss_roc[j].append(np.concatenate((y_cv1_loss[j][i], y_cv1_loss[j][0])))
                y_cv1_loss_roc_sig[i].append(np.concatenate((y_cv1_loss[j][i], y_cv1_loss[j][0])))
                y_cv2_loss_roc[j].append(np.concatenate((y_cv2_loss[j][i], y_cv2_loss[j][0])))
                y_cv2_loss_roc_sig[i].append(np.concatenate((y_cv2_loss[j][i], y_cv2_loss[j][0])))
        for i in range(len(model_names_short)):
            draw.plot_roc_curve(y_true, y_tea_loss_roc[i], signal_names, inputs, f"tea_{model_names_short[i]}")
            draw.plot_roc_curve(y_true, y_cv1_loss_roc[i], signal_names, inputs, f"cv1_{model_names_short[i]}")
            draw.plot_roc_curve(y_true, y_cv2_loss_roc[i], signal_names, inputs, f"cv2_{model_names_short[i]}")
        for i in range(len(signal_names)-1):
            draw.plot_roc_curve([y_true[i],y_true[i],y_true[i]], y_tea_loss_roc_sig[i+1], model_names_long, [inputs[i],inputs[i],inputs[i]], f"tea_signal_{signal_names[i+1]}")
            draw.plot_roc_curve([y_true[i],y_true[i],y_true[i]], y_cv1_loss_roc_sig[i+1], model_names_long, [inputs[i],inputs[i],inputs[i]], f"cv1_signal_{signal_names[i+1]}")
            draw.plot_roc_curve([y_true[i],y_true[i],y_true[i]], y_cv2_loss_roc_sig[i+1], model_names_long, [inputs[i],inputs[i],inputs[i]], f"cv2_signal_{signal_names[i+1]}")'''

        print("Finished plotting roc curves... plotting anomaly score distribution")
        # Declaring dicts to be used in anomaly score distribution
        for y_loss, name in zip([y_tea_loss, y_cv1_loss, y_cv2_loss], ["tea", "cv1", "cv2"]):
            results, results_scn, results_spr = dict(), dict(), dict()
            results_mod, results_sig = [dict(), dict(), dict()], [dict(), dict(), dict(), dict(), dict(), dict()]
            list_results_mod, list_results_sig = [], []
            for i in range(3):
                for j in range(len(signal_names)):
                    results_mod[i][f'{signal_names[j]}'] = y_loss[i][j]
                list_results_mod.append(list(results_mod[i].values()))
                draw.plot_anomaly_score_distribution(list_results_mod[i], [*results_mod[i]], f"anomaly_score_{name}_{model_names_short[i]}", xlim=[0, 256])
            for i in range(len(signal_names)):
                for j in range(3):
                    results_sig[i][f'{model_names_long[j]}'] = y_loss[j][i]
                list_results_sig.append(list(results_sig[i].values()))
                draw.plot_anomaly_score_distribution(list_results_sig[i], [*results_sig[i]], f"anomaly_score_{name}_signal_{signal_names[i]}", xlim=[0, 256])
        '''results_tea, results_tea_scn, results_tea_spr = dict(), dict(), dict()
        results_tea_mod, results_tea_sig = [dict(), dict(), dict()], [dict(), dict(), dict(), dict(), dict(), dict()]
        results_cv1, results_cv1_scn, results_cv1_spr = dict(), dict(), dict()
        results_cv1_mod, results_cv1_sig = [dict(), dict(), dict()], [dict(), dict(), dict(), dict(), dict(), dict()]
        results_cv2, results_cv2_scn, results_cv2_spr = dict(), dict(), dict()
        results_cv2_mod, results_cv2_sig = [dict(), dict(), dict()], [dict(), dict(), dict(), dict(), dict(), dict()]
        list_results_tea_mod, list_results_tea_sig = [], []
        list_results_cv1_mod, list_results_cv1_sig = [], []
        list_results_cv2_mod, list_results_cv2_sig = [], []
        for i in range(3):
            for j in range(len(signal_names)):
                results_tea_mod[i][f'{signal_names[j]}'] = y_tea_loss[i][j]
                results_cv1_mod[i][f'{signal_names[j]}'] = y_cv1_loss[i][j]
                results_cv2_mod[i][f'{signal_names[j]}'] = y_cv2_loss[i][j]
            list_results_tea_mod.append(list(results_tea_mod[i].values()))
            list_results_cv1_mod.append(list(results_cv1_mod[i].values()))
            list_results_cv2_mod.append(list(results_cv2_mod[i].values()))
            draw.plot_anomaly_score_distribution(list_results_tea_mod[i], [*results_tea_mod[i]], f"anomaly_score_tea_{model_names_short[i]}", xlim=[0,256])
            draw.plot_anomaly_score_distribution(list_results_cv1_mod[i], [*results_cv1_mod[i]], f"anomaly_score_cv1_{model_names_short[i]}", xlim=[0,256])
            draw.plot_anomaly_score_distribution(list_results_cv2_mod[i], [*results_cv2_mod[i]], f"anomaly_score_cv2_{model_names_short[i]}", xlim=[0,256])
        for i in range(len(signal_names)):
            for j in range(3):
                results_tea_sig[i][f'{model_names_long[j]}'] = y_tea_loss[j][i]
                results_cv1_sig[i][f'{model_names_long[j]}'] = y_cv1_loss[j][i]
                results_cv2_sig[i][f'{model_names_long[j]}'] = y_cv2_loss[j][i]
            list_results_tea_sig.append(list(results_tea_sig[i].values()))
            list_results_cv1_sig.append(list(results_cv1_sig[i].values()))
            list_results_cv2_sig.append(list(results_cv2_sig[i].values()))
            draw.plot_anomaly_score_distribution(list_results_tea_sig[i], [*results_tea_sig[i]], f"anomaly_score_tea_signal_{signal_names[i]}", xlim=[0,256])
            draw.plot_anomaly_score_distribution(list_results_cv1_sig[i], [*results_cv1_sig[i]], f"anomaly_score_cv1_signal_{signal_names[i]}", xlim=[0,256])
            draw.plot_anomaly_score_distribution(list_results_cv2_sig[i], [*results_cv2_sig[i]], f"anomaly_score_cv2_signal_{signal_names[i]}", xlim=[0,256])'''
        print("Finished plotting anomaly score distribution... plotting scatter plots and finding score correlations")
        # Anomaly score comparison statistics
        corr_tea, scatter_fit_tea, emd_tea = [[], []], [[], []], [[], []]
        corr_cv1, scatter_fit_cv1, emd_cv1 = [[], []], [[], []], [[], []]
        corr_cv2, scatter_fit_cv2, emd_cv2 = [[], []], [[], []], [[], []]
        for i in range(len(signal_names)):
            for j in range(2):
                #corr_tea[j].append(np.corrcoef(y_tea_loss[0][i], y_tea_loss[j+1][i]))
                #corr_cv1[j].append(np.corrcoef(y_cv1_loss[0][i], y_cv1_loss[j+1][i])) # this line takes forever when i = 1, j = 0
                #corr_cv2[j].append(np.corrcoef(y_cv2_loss[0][i], y_cv2_loss[j+1][i]))
                scatter_fit_tea[j].append(draw.plot_scatter_score_comparison(np.sqrt(y_tea_loss[0][i]), np.sqrt(y_tea_loss[j+1][i]), f"{model_names_long[0]} score (sqrt)", f"{model_names_long[j+1]} score (sqrt)", f"teacher_{model_names_short[0]}_{model_names_short[j+1]}_{signal_names[i]}", limits="equalsignal"))
                scatter_fit_cv1[j].append(draw.plot_scatter_score_comparison(np.sqrt(y_cv1_loss[0][i]), np.sqrt(y_cv1_loss[j+1][i]), f"{model_names_long[0]} score (sqrt)", f"{model_names_long[j+1]} score (sqrt)", f"cv1_{model_names_short[0]}_{model_names_short[j+1]}_{signal_names[i]}", limits="equalsignal"))
                scatter_fit_cv2[j].append(draw.plot_scatter_score_comparison(np.sqrt(y_cv2_loss[0][i]), np.sqrt(y_cv2_loss[j+1][i]), f"{model_names_long[0]} score (sqrt)", f"{model_names_long[j+1]} score (sqrt)", f"cv2_{model_names_short[0]}_{model_names_short[j+1]}_{signal_names[i]}", limits="equalsignal"))
                #emd_tea[j].append(wasserstein_distance(np.sqrt(y_tea_loss[0][i]), np.sqrt(y_tea_loss[j+1][i])))
                #emd_cv1[j].append(wasserstein_distance(np.sqrt(np.reshape(y_cv1_loss[0][i], (-1))), np.sqrt(np.reshape(y_cv1_loss[j+1][i], (-1)))))
                #emd_cv2[j].append(wasserstein_distance(np.sqrt(np.reshape(y_cv2_loss[0][i], (-1))), np.sqrt(np.reshape(y_cv2_loss[j+1][i], (-1)))))
        score_dist_tea = draw.plot_score_comparison_distributions(y_tea_loss, f"Difference", f"Frequency", "tea", limits="equalsignal")
        score_dist_cv1 = draw.plot_score_comparison_distributions(y_cv1_loss, f"Difference", f"Frequency", "cv1", limits="equalsignal")
        score_dist_cv2 = draw.plot_score_comparison_distributions(y_cv2_loss, f"Difference", f"Frequency", "cv2", limits="equalsignal")
        # score_dist is the bottleneck

        f = open(f"runs/{run_title}/summary.txt", "w")
        f.write(f"Trained on {X_train.shape[0]} events\n")
        for i in range(len(signal_names)):
            for j in range(2):
                f.write(f"tea_{model_names_short[j+1]}_{signal_names[i]}:\n")
                #f.write(f"corr: {corr_tea[j][i]}\n")
                f.write(f"M: {scatter_fit_tea[j][i][0]}\n")
                f.write(f"s: {score_dist_tea[j][i][0]}\n")
                f.write(f"mu: {score_dist_tea[j][i][1]}\n")
                #f.write(f"emd: {emd_tea[j][i]}\n")
                f.write(f"cv1_{model_names_short[j+1]}_{signal_names[i]}:\n")
                #f.write(f"corr: {corr_cv1[j][i]}\n")
                f.write(f"M: {scatter_fit_cv1[j][i][0]}\n")
                f.write(f"s: {score_dist_cv1[j][i][0]}\n")
                f.write(f"mu: {score_dist_cv1[j][i][1]}\n")
                #f.write(f"emd: {emd_cv1[j][i]}\n")
                f.write(f"cv2_{model_names_short[j+1]}_{signal_names[i]}:\n")
                #f.write(f"corr: {corr_cv2[j][i]}\n")
                f.write(f"M: {scatter_fit_cv2[j][i][0]}\n")
                f.write(f"s: {score_dist_cv2[j][i][0]}\n")
                f.write(f"mu: {score_dist_cv2[j][i][1]}\n")
                #f.write(f"emd: {emd_cv2[j][i]}\n")
        f.close()

    t4=time.time()

    f = open(f"runs/{run_title}/summary.txt", "w")
    f.write(f"Total time: {t4-t0}\n")
    f.write(f"Data/generator time: {t1-t0}\n")
    f.write(f"Teacher training time: {t2-t1}\n")
    f.write(f"Student training time: {t3-t2}\n")
    f.write(f"Evaluation time: {t4-t3}\n")
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
        "-dense1",
        "-dense1",
        type=int,
        nargs="+",
        help="dense1 for cv1. Accepts int.",
        default=[8],
    )
    parser.add_argument(
        "-dropout1",
        "-dropout1",
        type=float,
        nargs="+",
        help="dropout1 for cv1. Accepts float.",
        default=[0.1],
    )
    parser.add_argument(
        "-dense2",
        "-dense2",
        type=int,
        nargs="+",
        help="dense1 for cv2. Accepts int.",
        default=[8],
    )
    parser.add_argument(
        "-dropout2",
        "-dropout2",
        type=float,
        nargs="+",
        help="dropout1, 2 for cv2. Accepts floats.",
        default=[0.1, 0.1],
    )
    parser.add_argument(
        "-filters2",
        "-filters2",
        type=int,
        nargs="+",
        help="nFilters for cv2. Accepts int.",
        default=[2],
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
        "-evaluate",
        action="store_true",
        help="Evaluate models",
        default=False,
    )
    parser.add_argument(
        "-train-teachers",
        action="store_true",
        help="Train teacher models",
        default=False,
    )
    parser.add_argument(
        "-train-students",
        action="store_true",
        help="Train student models",
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
    run_training(run_title=args.run_title, config=config, train_teachers=args.train_teachers, train_students=args.train_students, eval=args.evaluate, epochs=args.epochs, data_to_use=args.data,
                 Lambda=args.Lambda, filters=args.filters, pooling=tuple(args.pool),
                 dense1=args.dense1, dropout1=args.dropout1, dense2=args.dense2, dropout2=args.dropout2, filters2=args.filters2,
                 search=args.search, verbose=args.verbose)


if __name__ == "__main__":
    main()
