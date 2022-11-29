import os
import yaml
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.model_selection import StratifiedKFold

from utils.data_loading import import_data
from utils.explainability import get_heatmap
from utils.helpers import create_logger, save_experiment

from models.mtex_cnn import mtex_cnn
from models.xcm import xcm
from models.xcm_seq import xcm_seq

import tensorflow as tf
from tensorflow import keras


if __name__ == "__main__":

    # Load configuration
    parser = argparse.ArgumentParser(description="XCM")
    parser.add_argument(
        "-c", "--config", default="configuration/config.yml", help="Configuration File"
    )
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        configuration = yaml.safe_load(config_file)

    if configuration["model_name"] in ["XCM", "XCM-Seq"]:
        window_size = configuration["window_size"]
    else:
        window_size = 0
    model_dict = {"XCM": xcm, "XCM-Seq": xcm_seq, "MTEX-CNN": mtex_cnn}

    # Create experiment folder
    xp_dir = (
        "./results/"
        + str(configuration["dataset"])
        + "/"
        + str(configuration["model_name"])
        + "/XP_"
        + str(configuration["experiment_run"])
        + "/"
    )
    save_experiment(xp_dir, args.config)
    log, logclose = create_logger(log_filename=os.path.join(xp_dir, "experiment.log"))
    log("Model: " + configuration["model_name"])

    # Load dataset
    (
        X_train,
        y_train,
        X_test,
        y_test,
        y_train_nonencoded,
        y_test_nonencoded,
    ) = import_data(configuration["dataset"], log)
    print("X_train.shape: ", X_train.shape)
    print("X_test.shape: ", X_test.shape)

    # Instantiate the cross validator
    skf = StratifiedKFold(
        n_splits=configuration["cv_folds"],
        random_state=configuration["random_state"],
        shuffle=True,
    )

    # Instantiate the result dataframes
    train_val_epochs_precisons = pd.DataFrame(
        columns=["Fold", "Epoch", "precison_Train", "precison_Validation"]
    )
    results = pd.DataFrame(
        columns=[
            "Dataset",
            "Model_Name",
            "Batch_Size",
            "Window_Size",
            "Fold",
            "precison_Train",
            "precison_Validation",
            "precison_Test",
        ]
    )

    # Loop through the indices the split() method returns
    for index, (train_indices, val_indices) in enumerate(
        skf.split(X_train, y_train_nonencoded)
    ):
        log("\nTraining on fold " + str(index + 1))

        # Generate batches from indices
        xtrain, xval = X_train[train_indices], X_train[val_indices]
        ytrain, yval, ytrain_nonencoded, yval_nonencoded = (
            y_train[train_indices],
            y_train[val_indices],
            y_train_nonencoded[train_indices],
            y_train_nonencoded[val_indices],
        )

        # Train the model
        if configuration["model_name"] in ["XCM", "XCM-Seq"]:
            model = model_dict[configuration["model_name"]](
                input_shape=X_train.shape[1:],
                n_class=y_train.shape[1],
                window_size=configuration["window_size"],
            )
        else:
            model = model_dict[configuration["model_name"]](
                input_shape=X_train.shape[1:], n_class=y_train.shape[1]
            )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=keras.metrics.Precision(name='precision')
        )
        h = model.fit(
            xtrain,
            ytrain,
            epochs=configuration["epochs"],
            batch_size=configuration["batch_size"],
            verbose=1,
            validation_data=(xval, yval),
        )

        # Calculate precisons
        fold_epochs_precisons = np.concatenate(
            (
                pd.DataFrame(np.repeat(index + 1, configuration["epochs"])),
                pd.DataFrame(range(1, configuration["epochs"] + 1)),
                pd.DataFrame(h.history["precision"]),
                pd.DataFrame(h.history["val_precision"]),
            ),
            axis=1,
        )
        prec_train = precision_score(
            ytrain_nonencoded, np.argmax(model.predict(xtrain), axis=1)
        )
        prec_val = precision_score(
            yval_nonencoded, np.argmax(model.predict(xval), axis=1)
        )
        prec_test = precision_score(
            y_test_nonencoded, np.argmax(model.predict(X_test), axis=1)
        )

        # Add fold results to the dedicated dataframe
        train_val_epochs_precisons = pd.concat(
            [
                train_val_epochs_precisons,
                pd.DataFrame(
                    fold_epochs_precisons,
                    columns=["Fold", "Epoch", "precison_Train", "precison_Validation"],
                ),
            ],
            axis=0,
        )
        results.loc[index] = [
            configuration["dataset"],
            configuration["model_name"],
            configuration["batch_size"],
            int(configuration["window_size"] * 100),
            index + 1,
            prec_train,
            prec_val,
            prec_test,
        ]
        log("precison Test: {0}".format(prec_test))

    # Train the model on the full train set
    log("\nTraining on the full train set")
    if configuration["model_name"] in ["XCM", "XCM-Seq"]:
        model = model_dict[configuration["model_name"]](
            input_shape=X_train.shape[1:],
            n_class=y_train.shape[1],
            window_size=configuration["window_size"],
        )
    else:
        model = model_dict[configuration["model_name"]](
            input_shape=X_train.shape[1:], n_class=y_train.shape[1]
        )
    print(model.summary())
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=keras.metrics.Precision(name='precision')
    )
    model.fit(
        X_train,
        y_train,
        epochs=configuration["epochs"],
        batch_size=configuration["batch_size"],
        verbose=1,
    )

    # Add result to the results dataframe
    prec_test = precision_score(
        y_test_nonencoded, np.argmax(model.predict(X_test), axis=1)
    )
    results["precison_Test_Full_Train"] = prec_test
    log("precison Test: {0}".format(prec_test))

    # Export model and results
    model.save(xp_dir + "/model.h5")
    train_val_epochs_precisons.to_csv(
        xp_dir + "/train_val_precisons.csv", index=False
    )
    results.to_csv(xp_dir + "/results.csv", index=False)
    print(results)

    # Example of a heatmap from Grad-CAM for the first MTS of the test set
    get_heatmap(
        configuration,
        xp_dir,
        model,
        X_train,
        X_test,
        y_train_nonencoded,
        y_test_nonencoded,
    )

    logclose()
