%% Load data and create train-test sets
cd DATA
breast_cancer_data = readtable('/Users/mariacarolina/Documents/IST/PYFUME_TUTORIAL/NN_SKLEARN/wbco.csv');
cd ..
X = table2array(breast_cancer_data(:,2:9));
Y = table2array(breast_cancer_data(:,10));
rng(4797);
train_test_partition = cvpartition(Y,'Holdout',0.2,'Stratify',true);
train_idx = training(train_test_partition);
test_idx = test(train_test_partition);
X_train = X(train_idx,:);
X_test = X(test_idx,:);
Y_train = Y(train_idx,:);
Y_test = Y(test_idx,:);

% Remove rows from X_train and Y_train where X_train contains NaN or Inf
invalid_idx_X = any(isnan(X_train), 2) | any(isinf(X_train), 2);
X_train(invalid_idx_X, :) = [];
Y_train(invalid_idx_X, :) = [];

% Remove rows from Y_train where Y_train contains NaN or Inf
invalid_idx_Y = isnan(Y_train) | isinf(Y_train);
X_train(invalid_idx_Y, :) = [];
Y_train(invalid_idx_Y, :) = [];

%% Apply Recursive Feature Elimination (RFE)
% Use RFE to find the optimal feature subset
mdl = fitrlinear(X_train, Y_train);  % Use a linear model for RFE
opts = statset('display', 'iter');
[~, history] = sequentialfs(@(train_X, train_Y, test_X, test_Y) ...
    loss(fitrlinear(train_X, train_Y), test_X, test_Y), ...
    X_train, Y_train, 'options', opts);

% Get the final selected features and their indices
selected_features_rfe = find(history.In(end,:));
X_train_rfe = X_train(:, selected_features_rfe);
X_test_rfe = X_test(:, selected_features_rfe);

% Display the number of features and their indices
fprintf('Number of features selected by RFE: %d\n', length(selected_features_rfe));
fprintf('Selected feature indices from the original dataset: ');
disp(selected_features_rfe);

% Train model on RFE-selected features
ts_model_rfe = genfis(X_train_rfe, Y_train, opt);

%% Check performance of model trained with RFE-selected features
Y_pred_rfe = evalfis(ts_model_rfe, X_test_rfe);
Y_pred_rfe(Y_pred_rfe>=0.5) = 1;
Y_pred_rfe(Y_pred_rfe<0.5) = 0;

class_report_rfe = classperf(Y_test, Y_pred_rfe);
fprintf('RFE Feature Model Accuracy: %4.3f \n', class_report_rfe.CorrectRate);
fprintf('RFE Feature Model Sensitivity: %4.3f \n', class_report_rfe.Sensitivity);
fprintf('RFE Feature Model Specificity: %4.3f \n', class_report_rfe.Specificity);
