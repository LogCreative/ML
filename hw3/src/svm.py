from sklearn import svm
from utils import *

def train_svm(dataset_name, feat_size):
    """Train SVM with normalized data.

    Args:
        dataset_name (str): the dataset name
        feat_size (int): the feature size
    """
    train_feat, train_label = loadData('data/{}.dat'.format(dataset_name), feat_size)
    model = svm.SVC(kernel='linear')
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

def run_svm(dataset_name, feat_size):
    """Run SVM with normalized data.

    Args:
        dataset_name (str): the dataset name
        feat_size (int): the feature size
    """
    model = train_svm(dataset_name, feat_size)
    acc = test_svm(model, dataset_name + '.t', feat_size)
    print("Dataset {} SVM Test Accuracy: {}".format(dataset_name, acc))

# run_svm("leu",7129)
run_svm("madelon",500)
run_svm("ijcnn1",22)