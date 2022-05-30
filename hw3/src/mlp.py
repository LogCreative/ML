from sklearn.neural_network import MLPClassifier
from utils import *

def train_mlp(dataset_name, feat_size):
    """Train MLP.

    Args:
        dataset_name (str): the dataset name
        feat_size (int): the feature size
    """
    train_feat, train_label = loadData('data/{}.dat'.format(dataset_name), feat_size)
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(int(feat_size*5)), random_state=1, max_iter=feat_size*30)
    model.fit(train_feat, train_label)
    return model

def test_mlp(model, dataset_name, feat_size):
    """Test MLP.

    Args:
        dataset_name (str): the dataset name
        feat_size (int): the feature size
    """
    test_feat, test_label = loadData('data/{}.dat'.format(dataset_name), feat_size)
    pred = model.predict(test_feat)
    return np.sum(pred == test_label) / len(test_label)

def run_mlp(dataset_name, feat_size):
    """Run MLP.

    Args:
        dataset_name (str): the dataset name
        feat_size (int): the feature size
    """
    start_time = time.time()
    model = train_mlp(dataset_name, feat_size)
    acc = test_mlp(model, dataset_name + '.t', feat_size)
    elapsed_time = time.time() - start_time
    print("Dataset: {} | MLP Test Accuracy: {} | Training Time: {}".format(dataset_name, acc, elapsed_time))

run_mlp("leu",7129)
run_mlp("madelon",500)
run_mlp("ijcnn1",22)
