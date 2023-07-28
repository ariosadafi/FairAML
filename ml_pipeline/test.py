import copy
import os.path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from dataset import *  # dataset
from model import *  # actual MIL model
from sklearn import metrics as metrics
import csv


kind = "sex"

CLASSES = ['control', 'RUNX1_RUNX1T1', 'NPM1', 'CBFB_MYH11', 'PML_RARA']

SOURCE_FOLDER = r'/Users/ario.sadafi/Desktop/dump4/TCIA_data_prepared'
TARGET_FOLDER = r"./output/"

patients = {}
with open(os.path.join(SOURCE_FOLDER,'metadata.csv'), newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        # print(line)
        patients[line[0]] = [os.path.join(SOURCE_FOLDER,
                                         "data",
                                         line[3],
                                         line[0],
                                         "fnl34_bn_features_layer_7.npy"), line[3]]


with open("../sexTestset.dat", "rb") as f:
    mTestset, fTestset = pickle.load(f)

Test_results = {}

def equalized_opportunity_multi_class(y_true, y_pred, protected_group):
    """
    Calculate the multi-class Equalized Opportunity for a given protected group.

    Parameters:
        y_true (numpy.array): Ground truth labels with shape (n_samples, n_classes).
        y_pred (numpy.array): Model predictions (probabilities) with shape (n_samples, n_classes).
        protected_group (numpy.array): A binary vector indicating the protected group with shape (n_samples,).

    Returns:
        equalized_opportunity (dict): A dictionary containing Equalized Opportunity for each class.
    """
    n_classes = y_true.shape[1]
    equalized_opportunity = {}

    for class_idx in range(n_classes):
        true_positive_group = np.logical_and(protected_group, y_true[:, class_idx] == 1)
        true_negative_group = np.logical_and(protected_group, y_true[:, class_idx] == 0)
        positive_predictions = np.logical_and(protected_group, y_pred[:, class_idx] >= 0.5)
        negative_predictions = np.logical_and(protected_group, y_pred[:, class_idx] < 0.5)

        true_positive_rate_diff = np.mean(positive_predictions) - np.mean(true_positive_group)
        false_positive_rate_diff = np.mean(negative_predictions) - np.mean(true_negative_group)

        equalized_opportunity[class_idx] = {
            'true_positive_rate_diff': true_positive_rate_diff,
            'false_positive_rate_diff': false_positive_rate_diff
        }

    return equalized_opportunity

def theil_index_multi_class(y_true, y_pred, protected_groups):
    """
    Calculate the Theil Index for multi-class classification with respect to protected groups.

    Parameters:
        y_true (numpy.array): Ground truth labels with shape (n_samples,).
        y_pred (numpy.array): Model predictions (probabilities) with shape (n_samples, n_classes).
        protected_groups (numpy.array): A categorical vector indicating the protected group for each sample with shape (n_samples,).

    Returns:
        theil_index (dict): A dictionary containing the Theil Index for each class.
    """
    n_classes = y_pred.shape[1]
    theil_index = {}

    # Calculate overall proportion of positive predictions
    mu = np.mean(y_pred)

    for class_idx in range(n_classes):
        class_probs = y_pred[:, class_idx]
        theil_index_class = 0.0

        for group in np.unique(protected_groups):
            group_mask = protected_groups == group
            group_mask = np.squeeze(group_mask)
            xi = np.mean(class_probs[group_mask])
            theil_index_class += (xi / mu) * np.log(xi / mu)

        theil_index_class /= len(np.unique(protected_groups))
        theil_index[class_idx] = theil_index_class

    return theil_index

def classwise_area_under_pr(predictions, ground_truth, num_classes):
    classwise_auc = np.zeros(num_classes)

    for class_index in range(num_classes):
        # Extract the predictions and ground truth for the current class
        y_pred_class = predictions[:, class_index]
        y_true_class = ground_truth[:, class_index]

        # Compute precision-recall curve
        precision, recall, _ = metrics.precision_recall_curve(y_true_class, y_pred_class)

        # Compute area under the PR curve for the current class
        classwise_auc[class_index] = metrics.auc(recall, precision)

    return classwise_auc
def calculate_metrics(pred,gt):
    pred = np.array(pred).squeeze()
    gt = np.array(gt)
    pred0 = np.argmax(pred, axis=1)
    gt0 = np.argmax(gt, axis=1)
    acc = metrics.accuracy_score(gt0, pred0)

    auc_pr = classwise_area_under_pr(pred, gt, 5)

    f1 = metrics.f1_score(gt0, pred0, average="macro")
    # print("Area under the PR curve for each class:", auc_pr)

    return acc, auc_pr, f1

def correct_order_classes(converter,auc):
    out = np.zeros(len(CLASSES))

    for i in range(len(CLASSES)):
        out[i] = auc[converter[CLASSES[i]]]

    return out


for exp in range(5):
    for fld in range(5):
        expname = kind + str(exp) + "f" + str(fld)


        class_converter = {}

        with open(os.path.join(TARGET_FOLDER, expname, 'class_conversion.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            for line in reader:
                class_converter[line[1]] = int(line[0])



        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        # load best performing model, and launch on test set
        model = torch.load(os.path.join(TARGET_FOLDER, expname, "state_dictmodel.pt"),map_location="cpu")
        model = model.to(device)

        eop_pred = []
        eop_gt = []
        eop_group = []

        pred = []
        gt = []
        for p in tqdm(mTestset):
            path, lbl_name = patients[p]
            lbl = np.zeros(5)
            lbl[class_converter[lbl_name]] = 1

            bag = np.load(path)

            bag = torch.tensor(bag).to(device)
            bag = torch.unsqueeze(bag,0)
            p, _, _, _ = model(bag)
            pred.append(F.softmax(p,dim=1).cpu().detach().numpy())
            gt.append(lbl)

        eop_pred.extend(copy.deepcopy(pred))
        eop_gt.extend(copy.deepcopy(gt))
        eop_group.extend(np.zeros(len(gt)))


        macc, mauc_pr, mf1 = calculate_metrics(pred, gt)


        pred = []
        gt = []
        for p in tqdm(fTestset):
            path, lbl_name = patients[p]
            lbl = np.zeros(5)
            lbl[class_converter[lbl_name]] = 1

            bag = np.load(path)

            bag = torch.tensor(bag).to(device)
            bag = torch.unsqueeze(bag,0)
            p, _, _, _ = model(bag)
            pred.append(F.softmax(p,dim=1).cpu().detach().numpy())
            gt.append(lbl)

        facc, fauc_pr, ff1 = calculate_metrics(pred, gt)

        eop_pred.extend(copy.deepcopy(pred))
        eop_gt.extend(copy.deepcopy(gt))
        eop_group.extend(np.ones(len(gt)))

        eop_pred = np.array(eop_pred).squeeze()
        eop_gt = np.array(eop_gt)
        eop_group = np.array(eop_group)
        eop_group = np.expand_dims(eop_group, 1)
        eq_opp = equalized_opportunity_multi_class(eop_gt,eop_pred,eop_group)
        theil = theil_index_multi_class(eop_gt,eop_pred,eop_group)

        # print(macc, mauc_pr)
        # print(facc, fauc_pr)

        mauc_pr = correct_order_classes(class_converter, mauc_pr)
        fauc_pr = correct_order_classes(class_converter, fauc_pr)

        # print(macc, mauc_pr)
        # print(facc, fauc_pr)
        Test_results[expname] = [[macc,mf1],[facc,ff1],mauc_pr,fauc_pr,[eq_opp, theil]]

with open("test_results-new.pkl", "wb") as f:
    pickle.dump(Test_results, f)
