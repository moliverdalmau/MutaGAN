'''
​​© 2020-2022 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

This material may be only be used, modified, or reproduced by or for the U.S. Government pursuant to the license rights granted under the clauses at DFARS 252.227-7013/7014 or FAR 52.227-14. For any other permission, please contact the Office of Technology Transfer at JHU/APL.
'''
import os
import time
import datetime
import pandas as pd
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K

import diff_match_patch as dmp_module

from nltk.translate.bleu_score import corpus_bleu
from boostershot_utility_functions import basic_statistical_analysis, calculate_levenshtein, confusion_matrix_analysis,\
    define_generator, evaluate_model_quick, find_sneath_index_file, load_or_create_unmatching_files, \
    make_protein_visuals, plot_mutation_locations, plot_mutation_subs, preprocess_data, process_x_data,\
    process_y_data, split_n, train_paired




if __name__ == "__main__":
    num_words = 4500
    step_size = 1
    window_size = 3
    hidden_space_dim = 128
    embedding_dim = 250
    train_data_file = "../data/20190220_IG_influenza_GAN_test_weighted_paths_and_leaves_trainUniqueParentsDiffParentChild.csv"
    train_data_dup_file = "../data/20190220_IG_influenza_GAN_test_weighted_paths_and_leaves_trainUniqueParentsSameParentChild.csv"
    test_data_file = "../data/20190220_IG_influenza_GAN_test_weighted_paths_and_leaves_testUniqueParentsDiffParentChild.csv"
    test_data_dup_file = "../data/20190220_IG_influenza_GAN_test_weighted_paths_and_leaves_testUniqueParentsSameParentChild.csv"
    validation_file = "../data/test_set_mutagan_2018_2019.csv"

    raw_training_data_file = "20190220_IG_influenza_GAN_train_weighted_leaves.csv"
    raw_testing_data_file = '20190220_IG_influenza_GAN_test_weighted_paths_and_leaves.csv'
    unmatching_data_file = '../utilities/unmatchingsequencesdiff.npy'
    unmatching_child_file = "../utilities/unmatchingsequenceschild.npy"
    unmatching_parent_file = "../utilities/unmatchingsequencesparent.npy"

    bad_encoder_json = '../utilities/BADEncoderFromEarlyModel4500_250.json'
    bad_encoder_weights = '../utilities/BadEncoder_ForBadDataGeneration.h5'
    bad_decoder_json = '../utilities/BADDecoderFromEarlyModel4500_250.json'
    bad_decoder_weights = '../utilities/BadDecoder_ForBadDataGeneration.h5'

    tokenizer_file = '../utilities/TokenizerGANV3.5.pkl'

    proper_formatting_convention = {'start': {'Single Peptide': 1, 'HA1': 17, 'HA2': 346},
                                    'end': {'Single Peptide': 16, 'HA1': 345, 'HA2': 566}}

    bad_models = [[bad_encoder_json, bad_encoder_weights],
                  [bad_decoder_json, bad_decoder_weights]]

    proper_formatting_convention = pd.DataFrame.from_dict(proper_formatting_convention)

    tokenizer = preprocess_data(train_data_file, train_data_dup_file, test_data_file, test_data_dup_file, num_words,
                                tokenizer_file)

    # dictionary for converting numbers into amino acid threemer
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    X_test = pd.read_csv(test_data_file)
    X_train = pd.read_csv(train_data_file)
    X_train_dup = pd.read_csv(train_data_dup_file)
    X_train = pd.concat([X_train, X_train_dup])
    X_valid = pd.read_csv(validation_file)

    # Calculates the levenshtein distance and removes ones where the distances is greater than 10
    dmp = dmp_module.diff_match_patch()
    X_train['Diff'] = X_train.apply(lambda x: calculate_levenshtein(x), axis=1)
    X_test['Diff'] = X_test.apply(lambda x: calculate_levenshtein(x), axis=1)
    X_valid['Diff'] = X_valid.apply(lambda x: calculate_levenshtein(x), axis=1)
    X_train = X_train.loc[X_train['Diff'] <= 10]
    X_test = X_test.loc[X_test['Diff'] <= 10]
    X_valid = X_valid.loc[X_valid['Diff'] <= 10]

    # Clean and process the data
    tokenized_x_train, _, maxlen = process_x_data(X_train, tokenizer, window_size, step_size)
    tokenized_y_train, goal_sequences_full_y = process_y_data(X_train, tokenizer, window_size, step_size, maxlen)

    tokenized_x_test, _ = process_x_data(X_test, tokenizer, window_size, step_size, maxlen)
    tokenized_y_test, goal_sequences_full_test_y = process_y_data(X_test, tokenizer, window_size, step_size, maxlen)

    tokenized_x_valid, _ = process_x_data(X_valid, tokenizer, window_size, step_size, maxlen)
    tokenized_y_valid, goal_sequences_full_valid_y = process_y_data(X_valid, tokenizer, window_size, step_size, maxlen)

    full_train = [tokenized_x_train, tokenized_y_train, goal_sequences_full_y, X_train.Diff, X_train]
    full_test = [tokenized_x_test, tokenized_y_test, goal_sequences_full_test_y, X_test.Diff, X_test]

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d %H%M%S")

    # table of the most frequently occuring children for each parent in the test dataset
    unique_test_parents = X_test[['ParentSequence']].drop_duplicates()
    unique_valid_parents = X_valid[['ParentSequence']].drop_duplicates()

    unmatching_sequences_child, unmatching_sequences_parent, unmatching_sequences_diff = load_or_create_unmatching_files(
        unmatching_data_file, unmatching_parent_file, unmatching_child_file, tokenized_x_train, tokenized_y_train,
        bad_models, hidden_space_dim = hidden_space_dim, reverse_word_map = reverse_word_map, step_size = step_size, window_size = window_size)

    # Pretraining the autoencoder
    autoencoder_file_name = "../models/pretrainedInfluenza_biLSTM_autoencoder_model_" + str(
        hidden_space_dim) + "_" + str(
        num_words) + "_weightsV3.h5"
    encoder_file_name = "../models/pretrainedInfluenza_biLSTM_encoder_model_" + str(hidden_space_dim) + "_" + str(
        num_words) + "_weightsV3.h5"
    decoder_file_name = "../models/pretrainedInfluenza_biLSTM_decoder_model_" + str(hidden_space_dim) + "_" + str(
        num_words) + "_weightsV3.h5"

    K.clear_session()

    autoencoder, infenc, infdec = define_generator(hidden_space_dim, num_words, embedding_dim)
    autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    checkpoint = ModelCheckpoint(autoencoder_file_name, monitor='val_acc', save_weights_only=False, verbose=1,
                                 save_best_only=True, mode='max')
    autoencoder.fit([tokenized_x_train, np.zeros((tokenized_x_train.shape[0], hidden_space_dim * 2)),
                     np.zeros((tokenized_y_train.shape[0], hidden_space_dim * 2)), goal_sequences_full_y.astype('int')],
                    np.expand_dims(tokenized_x_train, -1), batch_size=15, epochs=1,
                    validation_data=([tokenized_x_test, np.zeros((tokenized_x_test.shape[0], hidden_space_dim * 2)),
                                      np.zeros((tokenized_y_test.shape[0], hidden_space_dim * 2)),
                                      goal_sequences_full_test_y.astype('int')], np.expand_dims(tokenized_x_test, -1)),
                    callbacks=[checkpoint])

    autoencoder.layers[1].trainable = False
    autoencoder.layers[2].trainable = False
    autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    autoencoder.fit([tokenized_x_train, np.zeros((tokenized_x_train.shape[0], hidden_space_dim * 2)),
                     np.zeros((tokenized_x_train.shape[0], hidden_space_dim * 2)), goal_sequences_full_y.astype('int')],
                    np.expand_dims(tokenized_x_train, -1), batch_size=15, epochs=1,
                    validation_data=([tokenized_x_test, np.zeros((tokenized_x_test.shape[0], hidden_space_dim * 2)),
                                      np.zeros((tokenized_x_test.shape[0], hidden_space_dim * 2)),
                                      goal_sequences_full_test_y.astype('int')], np.expand_dims(tokenized_x_test, -1)),
                    callbacks=[checkpoint])
    K.clear_session()
    autoencoder, infenc, infdec = define_generator(hidden_space_dim, num_words, embedding_dim)
    autoencoder.load_weights(autoencoder_file_name)

    infenc.save_weights(encoder_file_name)
    infdec.save_weights(decoder_file_name)
    infenc.load_weights(encoder_file_name)

    unmatching_files = [unmatching_child_file, unmatching_parent_file, unmatching_data_file]
    model_file_names = [encoder_file_name, decoder_file_name, autoencoder_file_name]
    d_losses = []
    g_losses = []
    losses = [d_losses, g_losses]
    _, _, _ = train_paired(full_train, full_test, unmatching_files, model_file_names,
                                                  hidden_space_dim, num_words, embedding_dim, losses, epochs=1, batch_size=16,
                                                  gen_loops=5, first_pass=True, remove_parent_to_parent_training=True, batch_count=20)
    d_losses, g_losses, file_names = train_paired(full_train, full_test, unmatching_files, model_file_names,
                                                  hidden_space_dim, num_words, embedding_dim, losses, epochs = 1, batch_size = 22,
                                                  gen_loops = 5, first_pass = False, remove_parent_to_parent_training = True, batch_count = 20)

    # EVALUATION
    encoder_file_name = file_names[0]
    decoder_file_name = file_names[1]

    autoencoder, infenc, infdec = define_generator(hidden_space_dim, num_words, embedding_dim)
    infdec.load_weights(decoder_file_name)
    infenc.load_weights(encoder_file_name)

    sequence_level_results_unk, parent_level_results_unk = evaluate_model_quick(infenc, infdec, unique_valid_parents,
                                                                                X_valid, reverse_word_map,
                                                                                num_words,
                                                                                step_size, tokenizer, maxlen,
                                                                                proper_formatting_convention, tries=100,
                                                                                window_size=3, hidden_space_dim=128,
                                                                                max_allowed_differences=60,
                                                                                batch_size=10,
                                                                                drop_unknowns=False)


    save_to_directory = "../models"
    parent_level_file_name = os.path.join(save_to_directory, "validation_parent_level_results_" + st + ".csv")
    parent_level_results_unk.to_csv(parent_level_file_name)
    sequence_level_file_name = os.path.join(save_to_directory, "validation_sequence_level_results_" + st + ".csv")
    sequence_level_results_unk.to_csv(sequence_level_file_name)

    #code commented out but if user wants to read in already saved predictions, use the following line
    #parent_level_results_unk, sequence_level_results_unk = read_in_predictions(parent_level_file_name, sequence_level_file_name)

    sneath = find_sneath_index_file("../utilities/Sneath Index Similarity.csv")
    file = open(os.path.join(save_to_directory, "basic validation statistics_" + st + '.txt'), 'w')
    basic_statistical_analysis(parent_level_results_unk, sequence_level_results_unk, sneath, save_to_directory, file)
    confusion_matrix_analysis(parent_level_results_unk, 'parent_level_results', "", sneath, save_to_directory, file)
    confusion_matrix_analysis(sequence_level_results_unk, 'sequence_level_results', "", sneath, save_to_directory, file)
    plot_mutation_locations(X_train, X_valid, sequence_level_results_unk, save_to_directory)
    train_validation_freq_diff, train_gen_freq_diff, validation_gen_freq_diff = plot_mutation_subs(X_train, X_valid,
                                                                                              sequence_level_results_unk,
                                                                                              save_to_directory)
    file.write("*************************************************************************************")
    file.write("Avg frequency diff training and generated: " + str(abs(train_gen_freq_diff).mean().mean()))
    file.write("Avg frequency diff training and generated:" + str(abs(validation_gen_freq_diff).mean().mean()))
    make_protein_visuals(X_train, X_valid, sequence_level_results_unk, save_to_directory, include_histogram=True)
    # Calculate the BLEU score for the generated sequences.
    sequence_level_results_unk['Split_ChildSequence'] = sequence_level_results_unk.ChildSequence.apply(
        lambda x: [split_n(x) for x in x])
    sequence_level_results_unk['Split_GenSequence'] = sequence_level_results_unk.PredictedSequence.apply(
        lambda x: split_n(x))
    hypotheses = sequence_level_results_unk['Split_GenSequence'].tolist()
    list_of_references = sequence_level_results_unk['Split_ChildSequence'].tolist()
    BLEU_score = corpus_bleu(list_of_references, hypotheses)

    print("Bleu score: " + str(BLEU_score))
    file.write("*************************************************************************************")
    file.write("Bleu score :" + str(BLEU_score))

    file.close()
