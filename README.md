# Projects
Implementation Details in Brief
=> Generate Bag of Words (BoW) for the Input Corpus.
=> Run machine learning classifiers (using 10 Folds Cross Validation) on the BoW model
generated in the above step and record the classification results in‘Classifier_Classfication_Report.txt’ files.
=> Generate annotated corpus ‘Annotated_Corpus.txt’ by extracting annotations (entities)
for pages using DBPedia Spotlight.
=> Obtain TF-IDF scores for various entities in the annotated corpus by using sklearn’s
TFIDFVectorizer.
=> Generate BoW model for the annotated corpus – BoW model based on entity URIs.
=> Run the same machine learning classifiers used earlier on the BoW model based on entity URIs
and record classification results
‘Classifier_Classfication_Report_Annotated_Corpus.txt’ files.
=> Run the same machine learning classifiers from above step on the BoW model based on entity
URIs along with feature selection (using sklearn’s LinearSVC) and record classification results
‘Classifier_Classfication_Report_Annotated_Corpus_with_Feature_Selection.txt’ files.
