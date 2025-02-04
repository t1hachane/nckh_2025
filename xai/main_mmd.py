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

    model_path = str(sys.argv[5])

    # do not input space when declare a list
    # [ 1 ,   2] should be [1,2]
    view_list = list(map(int, sys.argv[6].strip("[]").split(",")))

    # 2500
    num_epoch = int(sys.argv[7])

    # 1e-3
    lr = float(sys.argv[8])

    testonly = str(sys.argv[9])

    hidden_dim = list(map(int, sys.argv[10].strip("[]").split(",")))

    print_hyper = str(sys.argv[11])

    if (print_hyper == True):
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
                    = {model_path}
                - List Views
                    = {view_list}

                *
                - Num Epoch PostTrain
                    = {num_epoch}

                - Lr Classifier
                    = {lr}
                - Hidden dim
                    = {hidden_dim}

                *
                - Test Only
                    = {testonly}
            """
        )

    if "BRCA" in data_folder:
        num_class = 5
    if "GBM" in data_folder:
        num_class = 4

    model_dict = train_test(
        data_folder,
        view_list,
        num_class,
        lr, 
        num_epoch,
        rseed,
        model_path, testonly,
        hidden_dim=[1000]
    )