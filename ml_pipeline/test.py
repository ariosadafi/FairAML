import os.path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from dataset import *  # dataset
from model import *  # actual MIL model
from sklearn import metrics as metrics
import csv

CLASSES = ['control', 'RUNX1_RUNX1T1', 'NPM1', 'CBFB_MYH11', 'PML_RARA']

SOURCE_FOLDER = r'/Users/ario.sadafi/Desktop/dump4/TCIA_data_prepared'
TARGET_FOLDER = r"/Users/ario.sadafi/PycharmProjects/F_AML/ml_pipeline/output/"

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


with open("/Users/ario.sadafi/PycharmProjects/F_AML001/sexTestset.dat", "rb") as f:
    mTestset, fTestset = pickle.load(f)

Test_results = {}

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

    # print("Area under the PR curve for each class:", auc_pr)

    return acc, auc_pr

def correct_order_classes(converter,auc):
    out = np.zeros(len(CLASSES))

    for i in range(len(CLASSES)):
        out[i] = auc[converter[CLASSES[i]]]

    return out





for exp in range(5):
    for fld in range(5):
        expname = "exp"+str(exp)+"f"+str(fld)


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

        macc, mauc_pr = calculate_metrics(pred, gt)


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

        facc, fauc_pr = calculate_metrics(pred, gt)

        # print(macc, mauc_pr)
        # print(facc, fauc_pr)

        mauc_pr = correct_order_classes(class_converter, mauc_pr)
        fauc_pr = correct_order_classes(class_converter, fauc_pr)

        # print(macc, mauc_pr)
        # print(facc, fauc_pr)
        Test_results[expname] = [macc,facc,mauc_pr,fauc_pr]

with open("test_results.pkl", "wb") as f:
    pickle.dump(Test_results, f)