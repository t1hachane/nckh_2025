""" Example for MOGONET classification
"""

import sys
from train_test import train_test
from utils import save_model_dict


if __name__ == "__main__":
    rseed = int(sys.argv[1])
    data_folder = str(sys.argv[2])
    postfix_tr = str(sys.argv[3])
    postfix_te = str(sys.argv[4])

    saved_model_dict_folder = str(sys.argv[5])

    # do not input space when declare a list
    # [ 1 ,   2] should be [1,2]
    view_list = list(map(int, sys.argv[6].strip("[]").split(",")))

    # 500
    num_epoch_pretrain = int(sys.argv[7])

    # 2500
    num_epoch = int(sys.argv[8])

    # 1e-3
    lr_e_pretrain = float(sys.argv[9])

    # 5e-4
    lr_e = float(sys.argv[10])

    # 1e-3
    lr_c = float(sys.argv[11])

    bool_using_early_stopping = str(sys.argv[12]).lower() in [
        "true",
        "1",
        "t",
        "y",
        "yes",
    ]
    verbose = str(sys.argv[13]).lower() in ["true", "1", "t", "y", "yes"]
    print_hyper = str(sys.argv[14]).lower() in ["true", "1", "t", "y", "yes"]
    dim_he_list = list(map(int, sys.argv[15].strip("[]").split(",")))

    if (print_hyper):
        print(
            f"""
            Config:
                * Reproduce:
                - Random Seed
                    = {rseed}

                * Data
                - Data Folder
                    = {data_folder}
                - Data Train
                    = {postfix_tr}
                - Data Test
                    = {postfix_te}
                - Saved Model Loc
                    = {saved_model_dict_folder}
                - List Views
                    = {view_list}

                *
                - Num Epoch PreTrain
                    = {num_epoch_pretrain}
                - Num Epoch PostTrain
                    = {num_epoch}
                - Lr Encoder PreTrain
                    = {lr_e_pretrain}
                - Lr Encoder PostTrain
                    = {lr_e}
                - Lr Classifier PostTrain
                    = {lr_c}
                - Bool using early stopping = {bool_using_early_stopping}
            """
        )

    if "BRCA" in data_folder:
        num_class = 5
    if "GBM" in data_folder:
        num_class = 4
    if bool_using_early_stopping:
        patience = int(sys.argv[16])
    model_dict = train_test(
        data_folder,
        view_list,
        num_class,
        lr_e_pretrain,
        lr_e,
        lr_c,
        num_epoch_pretrain,
        num_epoch,
        rseed,
        postfix_tr,
        postfix_te,
        patience,
        verbose,
        dim_he_list
    )
    save_model_dict(saved_model_dict_folder, model_dict)
