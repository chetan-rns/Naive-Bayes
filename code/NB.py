import pandas as pd

# read the data
data = pd.read_csv('diabetes.csv')

# labels for discretization
labels = ['low', 'medium', 'high']

# Preprocessing
for j in data.columns[:-1]:
    mean = data[j].mean()
    data[j] = data[j].replace(0, mean)
    # Converting numerical data to nominal data
    # Classifying quantitative values into labels : low,medium and high
    data[j] = pd.cut(data[j], bins=len(labels), labels=labels)


# train test split
# Training with different split ratios
split_per = [80, 70, 60]

# Function to count the values wrt the given column name


def count(data, colname, label, target):
    condition = (data[colname] == label) & (data['Outcome'] == target)
    return len(data[condition])


# For each split ratio
for i in split_per:

    # result list to store predicted values
    predicted = []

    # dictionary to store probabilities
    probabilities = {0: {}, 1: {}}

    # calculate training length
    train_len = int((i*len(data))/100)

    # Split training and testing data
    train_X = data.iloc[:train_len, :]

    test_X = data.iloc[train_len+1:, :-1]
    test_y = data.iloc[train_len+1:, -1]

    # count total number of 0s and 1s
    count_0 = count(train_X, 'Outcome', 0, 0)
    count_1 = count(train_X, 'Outcome', 1, 1)

    prob_0 = count_0/len(train_X)
    prob_1 = count_1/len(train_X)

    # Train the model
    # For each class in the training data
    for j in train_X.columns[:-1]:

        probabilities[0][j] = {}
        probabilities[1][j] = {}

        # For each label in that class
        for k in labels:
            count_k_0 = count(train_X, j, k, 0)
            count_k_1 = count(train_X, j, k, 1)
            # Calculate probability for class j and label k
            probabilities[0][j][k] = count_k_0 / count_0
            probabilities[1][j][k] = count_k_1 / count_1

    # Test the model
    for row in range(0, len(test_X)):
        prod_0 = prob_0
        prod_1 = prob_1
        for feature in test_X.columns:
            prod_0 *= probabilities[0][feature][test_X[feature].iloc[row]]
            prod_1 *= probabilities[1][feature][test_X[feature].iloc[row]]

        # Predict the outcome
        if prod_0 > prod_1:
            predicted.append(0)
        else:
            predicted.append(1)

    # create confusion matrix
    tp, tn, fp, fn = 0, 0, 0, 0
    for j in range(0, len(predicted)):
        if predicted[j] == 0:
            if test_y.iloc[j] == 0:
                tp += 1
            else:
                fp += 1
        else:
            if test_y.iloc[j] == 1:
                tn += 1
            else:
                fn += 1

    # Calculate the accuracy based on the confusion matrix
    # (true_positive+true_negative)/length(Test Data)
    print('Accuracy for training length ' +
          str(i)+'% : ', ((tp+tn)/len(test_y))*100)
