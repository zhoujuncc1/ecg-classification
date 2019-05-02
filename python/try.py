from train_SVM import *

def main(multi_mode='ovo', winL=65, winR=65, do_preprocess=True, use_weight_class=True,
         maxRR=True, use_RR=True, norm_RR=True, compute_morph={''}, oversamp_method='', pca_k='', feature_selection='',
         do_cross_val='', C_value=0.001, gamma_value=0.0, reduced_DS=False, leads_flag=[1, 0]):
    print("Runing train_SVM.py!")

    db_path = '../mitdb/m_learning/'

    # Load train data
    [tr_features, tr_labels, tr_patient_num_beats] = load_mit_db('DS1', winL, winR, do_preprocess,
                                                                 maxRR, use_RR, norm_RR, compute_morph, db_path,
                                                                 reduced_DS, leads_flag)
    # Filename
    oversamp_features_pickle_name = create_oversamp_name(reduced_DS, do_preprocess, compute_morph, winL, winR, maxRR,
                                                         use_RR, norm_RR, pca_k)

    # Do oversampling
    oversamp_method = 'SMOTE_regular'
    tr_features, tr_labels = perform_oversampling(oversamp_method, db_path + 'oversamp/python_mit',
                                                  oversamp_features_pickle_name, tr_features, tr_labels)


    # Load Test data
    [eval_features, eval_labels, eval_patient_num_beats] = load_mit_db('DS2', winL, winR, do_preprocess,
                                                                       maxRR, use_RR, norm_RR, compute_morph, db_path,
                                                                       reduced_DS, leads_flag)
    if reduced_DS == True:
        np.savetxt('mit_db/' + 'exp_2_' + 'DS2_labels.csv', eval_labels.astype(int), '%.0f')
    else:
        np.savetxt('mit_db/' + 'DS2_labels.csv', eval_labels.astype(int), '%.0f')

    np.savez('paper_train.npz', x_train=tr_features, y_train=tr_labels)
    np.savez('paper_test.npz', x_test=eval_features, y_test=eval_labels)



if __name__ == '__main__':
    main(reduced_DS=False, maxRR=False, use_RR=False, norm_RR=False, compute_morph=['raw'])