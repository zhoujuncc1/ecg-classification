from train_SVM import *

def main(multi_mode='ovo', winL=90, winR=90, do_preprocess=True, use_weight_class=True,
         maxRR=True, use_RR=True, norm_RR=True, compute_morph={''}, oversamp_method='', pca_k='', feature_selection='',
         do_cross_val='', C_value=0.001, gamma_value=0.0, reduced_DS=False, leads_flag=[1, 0]):
    print("Runing train_SVM.py!")

    db_path = '/home/zhoujun/workspace/research/mitdb/m_learning/'

    # Load train data
    [tr_features, tr_labels, tr_patient_num_beats] = load_mit_db('DS1', winL, winR, do_preprocess,
                                                                 maxRR, use_RR, norm_RR, compute_morph, db_path,
                                                                 reduced_DS, leads_flag)

    # Load Test data
    [eval_features, eval_labels, eval_patient_num_beats] = load_mit_db('DS2', winL, winR, do_preprocess,
                                                                       maxRR, use_RR, norm_RR, compute_morph, db_path,
                                                                       reduced_DS, leads_flag)
    if reduced_DS == True:
        np.savetxt('mit_db/' + 'exp_2_' + 'DS2_labels.csv', eval_labels.astype(int), '%.0f')
    else:
        np.savetxt('mit_db/' + 'DS2_labels.csv', eval_labels.astype(int), '%.0f')


if __name__ == '__main__':
    main(reduced_DS=True, maxRR=False, use_RR=False, norm_RR=False, compute_morph=['raw'])