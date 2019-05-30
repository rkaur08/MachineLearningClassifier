from sklearn import svm
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score


def run_zero_r_classifier(X_train, X_test, y_train, y_test, output_file_name, feature_selection_enabled=False):
    # Create a pipeline of transforms
    if feature_selection_enabled:
        pipe = Pipeline(
            [('feature_selection', SelectFromModel(LinearSVC(loss='squared_hinge', penalty='l1', dual=False))),
             ('model', DummyClassifier(strategy="stratified"))])
    else:
        pipe = Pipeline([('model', DummyClassifier(strategy="stratified"))])
    param_grid = [{}]  # Optionally add parameters here

    # Stratified 10-fold Cross Validation
    grid_search = GridSearchCV(pipe,
                               param_grid,
                               cv=StratifiedKFold(n_splits=10, random_state=42).split(X_train, y_train))

    model = grid_search.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    save_classification_report(y_test, y_preds, 'Zero R', output_file_name)


def run_neural_net_classifier(X_train, X_test, y_train, y_test, output_file_name, feature_selection_enabled=False):
    # Create a pipeline of transforms
    if feature_selection_enabled:
        pipe = Pipeline(
            [('feature_selection', SelectFromModel(LinearSVC(loss='squared_hinge', penalty='l1', dual=False))),
             ('model', MLPClassifier(alpha=1))])
    else:
        pipe = Pipeline([('model', MLPClassifier(alpha=1))])
    param_grid = [{}]  # Optionally add parameters here

    # Stratified 10-fold Cross Validation
    grid_search = GridSearchCV(pipe,
                               param_grid,
                               cv=StratifiedKFold(n_splits=10, random_state=42).split(X_train, y_train))

    model = grid_search.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    save_classification_report(y_test, y_preds, 'Neural Net', output_file_name)


def run_adaboost_classifier(X_train, X_test, y_train, y_test, output_file_name, feature_selection_enabled=False):
    # Create a pipeline of transforms
    if feature_selection_enabled:
        pipe = Pipeline(
            [('feature_selection', SelectFromModel(LinearSVC(loss='squared_hinge', penalty='l1', dual=False))),
             ('model', AdaBoostClassifier())])
    else:
        pipe = Pipeline([('model', AdaBoostClassifier())])
    param_grid = [{}]  # Optionally add parameters here

    # Stratified 10-fold Cross Validation
    grid_search = GridSearchCV(pipe,
                               param_grid,
                               cv=StratifiedKFold(n_splits=10, random_state=42).split(X_train, y_train))

    model = grid_search.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    save_classification_report(y_test, y_preds, 'AdaBoost', output_file_name)


def run_random_forest_classifier(X_train, X_test, y_train, y_test, output_file_name, feature_selection_enabled=False):
    # Create a pipeline of transforms
    if feature_selection_enabled:
        pipe = Pipeline(
            [('feature_selection', SelectFromModel(LinearSVC(loss='squared_hinge', penalty='l1', dual=False))),
             ('model', RandomForestClassifier(max_depth=2, random_state=0))])
    else:
        pipe = Pipeline([('model', RandomForestClassifier(max_depth=2, random_state=0))])
    param_grid = [{}]  # Optionally add parameters here

    # Stratified 10-fold Cross Validation
    grid_search = GridSearchCV(pipe,
                               param_grid,
                               cv=StratifiedKFold(n_splits=10, random_state=42).split(X_train, y_train))

    model = grid_search.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    save_classification_report(y_test, y_preds, 'Zero R', output_file_name)


def run_logistic_regression_classifier(X_train, X_test, y_train, y_test, output_file_name,
                                       feature_selection_enabled=False):
    # Create a pipeline of transforms
    if feature_selection_enabled:
        pipe = Pipeline(
            [('feature_selection', SelectFromModel(LinearSVC(loss='squared_hinge', penalty='l1', dual=False))),
             ('select', SelectFromModel(LogisticRegression(class_weight='balanced', penalty="l1", C=0.01))),
             ('model', LogisticRegression(class_weight='balanced', penalty='l2'))])
    else:
        pipe = Pipeline([('select', SelectFromModel(LogisticRegression(class_weight='balanced', penalty="l1", C=0.01))),
                         ('model', LogisticRegression(class_weight='balanced', penalty='l2'))])
    param_grid = [{}]  # Optionally add parameters here

    # Stratified 10-fold Cross Validation
    grid_search = GridSearchCV(pipe,
                               param_grid,
                               cv=StratifiedKFold(n_splits=10, random_state=42).split(X_train, y_train))

    model = grid_search.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    save_classification_report(y_test, y_preds, 'Logistic Regression', output_file_name)


def run_linear_svm_classifier(X_train, X_test, y_train, y_test, output_file_name, feature_selection_enabled=False):
    # Create a pipeline of transforms
    if feature_selection_enabled:
        pipe = Pipeline(
            [('feature_selection', SelectFromModel(LinearSVC(loss='squared_hinge', penalty='l1', dual=False))),
             ('model', SVC(kernel="linear", C=0.025))])
    else:
        pipe = Pipeline([('model', svm.LinearSVC())])
    param_grid = [{}]  # Optionally add parameters here

    # Stratified 10-fold Cross Validation
    grid_search = GridSearchCV(pipe,
                               param_grid,
                               cv=StratifiedKFold(n_splits=10, random_state=42).split(X_train, y_train))

    model = grid_search.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    save_classification_report(y_test, y_preds, 'Linear SVM', output_file_name)


def run_naive_bayes_classifier(X_train, X_test, y_train, y_test, output_file_name, feature_selection_enabled=False):
    # Create a pipeline of transforms
    if feature_selection_enabled:
        pipe = Pipeline(
            [('feature_selection', SelectFromModel(LinearSVC(loss='squared_hinge', penalty='l1', dual=False))),
             ('model', GaussianNB())])
    else:
        pipe = Pipeline([('model', GaussianNB())])
    param_grid = [{}]  # Optionally add parameters here

    # Stratified 10-fold Cross Validation
    grid_search = GridSearchCV(pipe,
                               param_grid,
                               cv=StratifiedKFold(n_splits=10, random_state=42).split(X_train, y_train))

    model = grid_search.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    save_classification_report(y_test, y_preds, 'Multinomial Naive Bayes', output_file_name)


def run_multinomial_naive_bayes_classifier(X_train, X_test, y_train, y_test, output_file_name,
                                           feature_selection_enabled=False):
    # Create a pipeline of transforms
    if feature_selection_enabled:
        pipe = Pipeline(
            [('feature_selection', SelectFromModel(LinearSVC(loss='squared_hinge', penalty='l1', dual=False))),
             ('model', MultinomialNB())])
    else:
        pipe = Pipeline([('model', MultinomialNB())])
    param_grid = [{}]  # Optionally add parameters here

    # Stratified 10-fold Cross Validation
    grid_search = GridSearchCV(pipe,
                               param_grid,
                               cv=StratifiedKFold(n_splits=10, random_state=42).split(X_train, y_train))

    model = grid_search.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    save_classification_report(y_test, y_preds, 'Multinomial Naive Bayes', output_file_name)

    # # Pickle the model
    # joblib.dump(grid_search.best_estimator_, 'best_classifier.pkl')


def save_classification_report(y_test, y_preds, classifier_name, output_file_name):
    # Generate Classification Report
    report = classification_report(y_test, y_preds)
    accuracy = accuracy_score(y_test, y_preds)

    # Save the Classification Report
    with open(output_file_name, 'w+') as classification_report_file:
        classification_report_file.write(report)
        classification_report_file.write("\nAccuracy: " + str(round(accuracy, 2) * 100) + " %\n")
        print("> Saved Classification Report of " + classifier_name + " to '" + output_file_name + "'.")
