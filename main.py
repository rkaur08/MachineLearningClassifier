import os

import arff
import numpy as np
import pandas as pd
import spotlight
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import classifiers


def create_directory_if_not_exist(directory_name):
    """Creates directory if it does not exist already"""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


# Create output directory
output_directory_name = 'out'
create_directory_if_not_exist(output_directory_name)
# Create classification output directory
classification_directory = 'classification'
create_directory_if_not_exist(os.path.join(output_directory_name, classification_directory))
# Create vector files output directory
vector_directory = 'vector'
create_directory_if_not_exist(os.path.join(output_directory_name, vector_directory))


def remove_file_if_exists(file_name):
    """
    Remove existing file to have a clean and updated output
    """
    try:
        if os.path.isfile(file_name):
            os.remove(file_name)
    except OSError:
        pass


def get_corpus_as_data_frame(corpus_file_name):
    corpus_arff = arff.load(open(corpus_file_name, 'r'))
    corpus_df = pd.DataFrame(corpus_arff['data'])
    corpus_df.columns = ['text', 'class']
    print("> Number of pages in the input corpus: " + str(len(corpus_df)))
    # remove rows with no text
    empty_text = corpus_df[corpus_df.text.isin(['', ' '])]
    print("> Found " + str(len(empty_text)) + " pages with no content")
    corpus_df_filtered = corpus_df[corpus_df.text != ""]
    corpus_df_filtered = corpus_df_filtered[corpus_df_filtered.text != " "]
    print("> Number of pages considered for evaluation: " + str(len(corpus_df_filtered)))

    return corpus_df_filtered


def generate_bag_of_words_for_corpus(corpus, output_file_name):
    features = get_bag_of_words(corpus['text'].tolist())
    # Take a dump of discovered features
    remove_file_if_exists(os.path.join(output_directory_name, vector_directory, output_file_name))
    joblib.dump(features, os.path.join(output_directory_name, vector_directory, output_file_name))
    print("> Generated bag of words and saved to '"
          + os.path.join(output_directory_name, vector_directory, output_file_name) + "'.")


def tokenize(text):
    """
    Uses NLTK's tokenizer to tokenize text
    """
    return word_tokenize(text)


def get_bag_of_words(documents):
    """
    Generates bag of words as features
    """
    count_vectorizer = CountVectorizer(max_features=10000, tokenizer=tokenize)
    features = count_vectorizer.fit_transform(documents).toarray()
    return features


def run_machine_learning_classifiers_on_input_corpus(vectors_file, corpus):
    """
    Use Word Vectors and Labels from Corpus and Run ML Classifiers
    """
    print("> Loading previously generated corpus vectors...")
    vectors = joblib.load(os.path.join(output_directory_name, vector_directory, vectors_file))
    print("> Done loading previously generated corpus vectors.")

    X = vectors
    y = corpus['class']

    # Create a train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

    # Run Zero-R Classifier (called 'Dummy Classifier in skslearn)
    output_file_name = 'Zero_R_Classification_Report.txt'
    print("> Running Zero R Classifier...")
    remove_file_if_exists(os.path.join(output_directory_name, output_file_name))
    classifiers.run_zero_r_classifier(X_train, X_test, y_train, y_test,
                                      os.path.join(output_directory_name, classification_directory, output_file_name))

    # Run Random Forest Classifier
    output_file_name = 'Random_Forest_Classification_Report.txt'
    remove_file_if_exists(os.path.join(output_directory_name, output_file_name))
    print("> Running Random Forest Classifier...")
    classifiers.run_random_forest_classifier(X_train, X_test, y_train, y_test,
                                             os.path.join(output_directory_name, classification_directory,
                                                          output_file_name))

    # Run Logistic Regression Classifier
    output_file_name = 'LR_Classification_Report.txt'
    remove_file_if_exists(os.path.join(output_directory_name, output_file_name))
    print("> Running Logistic Regression Classifier...")
    classifiers.run_logistic_regression_classifier(X_train, X_test, y_train, y_test,
                                                   os.path.join(output_directory_name, classification_directory,
                                                                output_file_name))

    # Run Linear SVM Classifier
    output_file_name = 'Linear_SVM_Classification_Report.txt'
    remove_file_if_exists(os.path.join(output_directory_name, output_file_name))
    print("> Running Linear SVM Classifier...")
    classifiers.run_linear_svm_classifier(X_train, X_test, y_train, y_test,
                                          os.path.join(output_directory_name, classification_directory,
                                                       output_file_name))

    # Run Naive Bayes Classifier
    output_file_name = 'Naive_Bayes_Classification_Report.txt'
    remove_file_if_exists(os.path.join(output_directory_name, classification_directory, output_file_name))
    print("> Running Naive Bayes Classifier...")
    classifiers.run_naive_bayes_classifier(X_train, X_test, y_train, y_test,
                                           os.path.join(output_directory_name, classification_directory,
                                                        output_file_name))

    # Run Multinomial Naive Bayes Classifier
    output_file_name = 'Multinomial_Naive_Bayes_Classification_Report.txt'
    remove_file_if_exists(os.path.join(output_directory_name, classification_directory, output_file_name))
    print("> Running Multinomial Naive Bayes Classifier...")
    classifiers.run_multinomial_naive_bayes_classifier(X_train, X_test, y_train, y_test,
                                                       os.path.join(output_directory_name, classification_directory,
                                                                    output_file_name))

    # Run AdaBoost Classifier
    output_file_name = 'AdaBoost_Classification_Report.txt'
    remove_file_if_exists(os.path.join(output_directory_name, classification_directory, output_file_name))
    print("> Running AdaBoost Classifier...")
    classifiers.run_adaboost_classifier(X_train, X_test, y_train, y_test,
                                        os.path.join(output_directory_name, classification_directory,
                                                     output_file_name))

    # Run Neural Net Classifier
    output_file_name = 'Neural_Net_Classification_Report.txt'
    remove_file_if_exists(os.path.join(output_directory_name, classification_directory, output_file_name))
    print("> Running Neural Net Classifier...")
    classifiers.run_neural_net_classifier(X_train, X_test, y_train, y_test,
                                          os.path.join(output_directory_name, classification_directory,
                                                       output_file_name))


def generate_annotated_corpus(corpus, output_file_name):
    corpus_pages = corpus['text'].tolist()
    corpus_page_classes = corpus['class'].tolist()

    remove_file_if_exists(os.path.join(output_directory_name, output_file_name))
    with open(os.path.join(output_directory_name, output_file_name), 'a+', encoding='utf-8') as output_file:
        # write a header line
        output_file.write('text|class\n')
        ai_entity_count = 0
        not_ai_entity_count = 0
        for corpus_page, corpus_page_class in zip(corpus_pages, corpus_page_classes):
            try:
                annotations = spotlight.annotate('http://model.dbpedia-spotlight.org/en/annotate', corpus_page)
                uri_text = ""
                for annotation in annotations:
                    uri_text += annotation['URI'] + " "
                output_file.write(uri_text[:-1] + "|" + corpus_page_class + "\n")
                if corpus_page_class == "AI":
                    ai_entity_count = ai_entity_count + len(annotations)
                else:
                    not_ai_entity_count = not_ai_entity_count + len(annotations)

            except Exception as e:
                print(e)

        print("> Total number of entities in AI Category: " + str(ai_entity_count))
        print("> Total number of entities in NOT_AI Category: " + str(not_ai_entity_count))


def compute_tfidf_for_annotated_corpus(corpus_file_name, output_file_name):
    """
    Generate tfidf values
    """
    # read annotated corpus
    corpus = pd.read_csv(os.path.join(output_directory_name, corpus_file_name), delimiter="|", header=0,
                         encoding='utf-8')

    print("> Total number of pages with entities returned by DBpedia Spotlight: " + str(len(corpus)))

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    tfidf_result = tfidf_vectorizer.fit_transform(corpus['text'].tolist())

    scores = zip(tfidf_vectorizer.get_feature_names(), np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # remove existing file to have a clean and updated output
    remove_file_if_exists(os.path.join(output_directory_name, output_file_name))

    # save ngrams and their tfidf scores to file
    with open(os.path.join(output_directory_name, output_file_name), 'a+', encoding='utf-8') as output_file:
        for item in sorted_scores:
            output_file.write("{0:50} TFIDF_Score: {1}\n".format(item[0], item[1]))


def generate_bag_of_words_using_entity_uris_for_corpus(corpus_file_name, output_file_name):
    # read annotated corpus
    corpus = pd.read_csv(os.path.join(output_directory_name, corpus_file_name), delimiter="|", header=0,
                         encoding='utf-8')
    features = get_bag_of_words_using_entity_uris(corpus['text'].tolist())
    # Take a dump of discovered features
    remove_file_if_exists(os.path.join(output_directory_name, vector_directory, output_file_name))
    joblib.dump(features, os.path.join(output_directory_name, vector_directory, output_file_name))
    print("> Generated bag of words with entity uris and saved to '"
          + os.path.join(output_directory_name, vector_directory, output_file_name) + "'.")


def get_bag_of_words_using_entity_uris(documents):
    """
    Generates bag of words using entity uris as features
    """
    count_vectorizer = CountVectorizer(max_features=10000)
    features = count_vectorizer.fit_transform(documents).toarray()
    return features


def run_machine_learning_classifiers_on_annotated_corpus(vectors_file, corpus_file_name, feature_selection_enabled,
                                                         output_file_suffix):
    """
    Use Word Vectors and Labels from Corpus and Run ML Classifiers
    """
    # read annotated corpus
    corpus = pd.read_csv(os.path.join(output_directory_name, corpus_file_name), delimiter="|", header=0,
                         encoding='utf-8')
    print("> Loading previously generated annotated corpus (with entity-uris) vectors...")
    vectors = joblib.load(os.path.join(output_directory_name, vector_directory, vectors_file))
    print("> Done loading previously generated annotated corpus (with entity-uris) vectors.")

    X = vectors
    y = corpus['class']

    # Create a train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    # Run Zero-R Classifier (called 'Dummy Classifier in sklearn)
    output_file_name = 'Zero_R_Classification_Report' + output_file_suffix + ".txt"
    remove_file_if_exists(os.path.join(output_directory_name, classification_directory, output_file_name))
    print("> Running Zero R Classifier...")
    classifiers.run_zero_r_classifier(X_train, X_test, y_train, y_test,
                                      os.path.join(output_directory_name, classification_directory, output_file_name),
                                      feature_selection_enabled)

    # Run Random Forest Classifier
    output_file_name = 'Random_Forest_Classification_Report' + output_file_suffix + ".txt"
    remove_file_if_exists(os.path.join(output_directory_name, classification_directory, output_file_name))
    print("> Running Random Forest Classifier...")
    classifiers.run_random_forest_classifier(X_train, X_test, y_train, y_test,
                                             os.path.join(output_directory_name, classification_directory,
                                                          output_file_name),
                                             feature_selection_enabled)

    # Run Logistic Regression Classifier
    output_file_name = 'LR_Classification_Report' + output_file_suffix + ".txt"
    remove_file_if_exists(os.path.join(output_directory_name, classification_directory, output_file_name))
    print("> Running Logistic Regression Classifier...")
    classifiers.run_logistic_regression_classifier(X_train, X_test, y_train, y_test,
                                                   os.path.join(output_directory_name, classification_directory,
                                                                output_file_name),
                                                   feature_selection_enabled)

    # Run Linear SVM Classifier
    output_file_name = 'Linear_SVM_Classification_Report' + output_file_suffix + ".txt"
    remove_file_if_exists(os.path.join(output_directory_name, classification_directory, output_file_name))
    print("> Running Linear SVM Classifier...")
    classifiers.run_linear_svm_classifier(X_train, X_test, y_train, y_test,
                                          os.path.join(output_directory_name, classification_directory,
                                                       output_file_name),
                                          feature_selection_enabled)

    # Run Naive Bayes Classifier
    output_file_name = 'Naive_Bayes_Classification_Report' + output_file_suffix + ".txt"
    remove_file_if_exists(os.path.join(output_directory_name, classification_directory, output_file_name))
    print("> Running Naive Bayes Classifier...")
    classifiers.run_naive_bayes_classifier(X_train, X_test, y_train, y_test,
                                           os.path.join(output_directory_name, classification_directory,
                                                        output_file_name),
                                           feature_selection_enabled)

    # Run Multinomial Naive Bayes Classifier
    output_file_name = 'Multinomial_Naive_Bayes_Classification_Report' + output_file_suffix + ".txt"
    remove_file_if_exists(os.path.join(output_directory_name, classification_directory, output_file_name))
    print("> Running Multinomial Naive Bayes Classifier...")
    classifiers.run_multinomial_naive_bayes_classifier(X_train, X_test, y_train, y_test,
                                                       os.path.join(output_directory_name, classification_directory,
                                                                    output_file_name),
                                                       feature_selection_enabled)

    # Run AdaBoost Classifier
    output_file_name = 'AdaBoost_Classification_Report' + output_file_suffix + ".txt"
    remove_file_if_exists(os.path.join(output_directory_name, classification_directory, output_file_name))
    print("> Running AdaBoost Classifier...")
    classifiers.run_adaboost_classifier(X_train, X_test, y_train, y_test,
                                        os.path.join(output_directory_name, classification_directory,
                                                     output_file_name),
                                        feature_selection_enabled)

    # Run Neural Net Classifier
    output_file_name = 'Neural_Net_Classification_Report' + output_file_suffix + ".txt"
    remove_file_if_exists(os.path.join(output_directory_name, classification_directory, output_file_name))
    print("> Running Neural Net Classifier...")
    classifiers.run_neural_net_classifier(X_train, X_test, y_train, y_test,
                                          os.path.join(output_directory_name, classification_directory,
                                                       output_file_name),
                                          feature_selection_enabled)


if __name__ == '__main__':
    print("##### PROGRAM START #####\n")

    # Read Corpus
    print("Reading the corpus...")
    corpus = get_corpus_as_data_frame(corpus_file_name='.arff')
    print("Done loading the corpus.\n")

    # Generate bag of words
    print("Generating bag of words for the Input Corpus...")
    generate_bag_of_words_for_corpus(corpus=corpus, output_file_name='word_vectors.pkl')
    print("Done generating bag of words vectors for the Input Corpus.\n")

    # Run Classifier on Input Corpus
    print("Running Machine Learning Classifiers on the Corpus...")
    run_machine_learning_classifiers_on_input_corpus(vectors_file='word_vectors.pkl', corpus=corpus)
    print("Done Running all the Machine Learning Classifiers.\n")

    # Generate Annotated Corpus
    print("Generating annotated corpus...")
    generate_annotated_corpus(corpus=corpus, output_file_name='Annotated_Corpus.txt')
    print("Generated and saved annotated corpus to 'Annotated_Corpus.txt'.\n")

    # Generate TF-IDF
    print("Generating TF-IDF for the annotated corpus...")
    compute_tfidf_for_annotated_corpus(corpus_file_name='Annotated_Corpus.txt',
                                       output_file_name='annotated_corpus_tfidf.txt')
    print("Generated and saved TF-IDF for the annotated corpus to 'annotated_corpus_tfidf'.")

    # Generate bag of words
    print("Generating bag of words using Entity URIs for the Annotated Corpus...")
    generate_bag_of_words_using_entity_uris_for_corpus(corpus_file_name='Annotated_Corpus.txt',
                                                       output_file_name='word_vectors_using_entity_uris.pkl')
    print("Generated bag of words using Entity URIs and saved to 'word_vectors_using_entity_uris.pkl'.\n")

    # Run Classifiers on Annotated Corpus (URLs instead of text)
    print("Running Machine Learning Classifiers on the Annotated Corpus...")
    run_machine_learning_classifiers_on_annotated_corpus(vectors_file='word_vectors_using_entity_uris.pkl',
                                                         corpus_file_name='Annotated_Corpus.txt',
                                                         feature_selection_enabled=False,
                                                         output_file_suffix='_Annotated_Corpus')
    print("Done Running all the Machine Learning Classifiers on the Annotated Corpus.\n")

    # Run Classifiers on Annotated Corpus with Feature Selection
    print("Running Machine Learning Classifiers on the Annotated Corpus with Feature Selection...")
    run_machine_learning_classifiers_on_annotated_corpus(vectors_file='word_vectors_using_entity_uris.pkl',
                                                         corpus_file_name='Annotated_Corpus.txt',
                                                         feature_selection_enabled=True,
                                                         output_file_suffix='_Annotated_Corpus_with_Feature_Selection')
    print("Done Running all the Machine Learning Classifiers on the Annotated Corpus with Feature Selection.\n")

    print("\n##### PROGRAM STOP #####")
