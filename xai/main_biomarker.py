import sys
import os
import copy
from feat_importance import cal_feat_imp, summarize_imp_feat

if __name__ == "__main__":
    data_folder = str(sys.argv[1])
    model_folder = str(sys.argv[2])
    dim_he_list = list(map(int, sys.argv[3].strip("[]").split(",")))
    view_list = list(map(int, sys.argv[4].strip("[]").split(",")))
    num_models = int(sys.argv[5])
    postfix_tr = str(sys.argv[6])
    postfix_te = str(sys.argv[7])

    mogonet_top_biomarkers_folder = str(sys.argv[8])
    biomarker_file_name = str(sys.argv[9])
    topn = int(sys.argv[10])

    if "BRCA" in data_folder:
        num_class = 5
    if "GBM" in data_folder:
        num_class = 4
    if "Lung" in data_folder:
        num_class = 2
    if "CRC" in data_folder:
        num_class = 4

    featimp_list_list = []
    for rep in range(num_models):
        featimp_list = cal_feat_imp(
            os.path.join(data_folder, str(rep + 1)),
            os.path.join(model_folder, str(rep + 1)),
            dim_he_list,
            view_list,
            num_class,
            postfix_tr,
            postfix_te,
        )
        featimp_list_list.append(copy.deepcopy(featimp_list))
    summarize_imp_feat(
        featimp_list_list=featimp_list_list,
        mogonet_top_biomarkers_folder=mogonet_top_biomarkers_folder,
        biomarker_file_name=biomarker_file_name,
        topn=topn
    )