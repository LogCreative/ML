from sklearn import svm
from utils import *

def train_svm(dataset_name, feat_size, kernel='linear', C=1):
    """Train SVM with normalized data.

    Args:
        dataset_name (str): the dataset name
        feat_size (int): the feature size
    """
    train_feat, train_label = loadData('data/{}.dat'.format(dataset_name), feat_size)
    model = svm.SVC(kernel=kernel, C=C)
    model.fit(train_feat, train_label)
    return model

def test_svm(model, dataset_name, feat_size):
    """Test SVM with normalized data.

    Args:
        model (SVM): the trained SVM model
        dataset_name (str): the dataset name
        feat_size (int): the feature size
    """
    test_feat, test_label = loadData('data/{}.dat'.format(dataset_name), feat_size)
    pred = model.predict(test_feat)
    return np.sum(pred == test_label) / len(test_label)

def run_svm(dataset_name, feat_size, kernel='linear', C=1):
    """Run SVM with normalized data.

    Args:
        dataset_name (str): the dataset name
        feat_size (int): the feature size
    """
    start_time = time.time()
    model = train_svm(dataset_name, feat_size, kernel, C)
    acc = test_svm(model, dataset_name + '.t', feat_size)
    elapsed_time = time.time() - start_time
    print("Dataset: {} | C={}--{} SVM Test Accuracy: {} | Training Time: {}".format(dataset_name, C, kernel, acc, elapsed_time))

for k in ['linear', 'poly', 'rbf', 'sigmoid']:
    run_svm('madelon', 500, k)
    run_svm('ijcnn1', 22, k)

for c in [0.01, 0.1, 0.5, 1]:
    run_svm('madelon', 500, C=c)
    run_svm('ijcnn1', 22, C=c)