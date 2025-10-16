import numpy as np
from sklearn import metrics


def metric_log(test_preds, test_label):
    '''
        主要是使用 sklearn 计算分类问题的评价指标

    '''
    matrix = metrics.confusion_matrix(test_label, test_preds)
    classification = metrics.classification_report(test_label, test_preds, digits=4)
    accuracy = metrics.accuracy_score(test_label, test_preds)
    precision = metrics.precision_score(test_label, test_preds, average='macro')
    recall = metrics.recall_score(test_label, test_preds, average='macro')
    f1 = metrics.f1_score(test_label, test_preds, average='macro')
    kappa = metrics.cohen_kappa_score(test_label, test_preds)
    # fpr, tpr, thresholds = metrics.roc_curve(test_label, test_score, drop_intermediate=True)
    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
    #                                   estimator_name='example estimator')
    # display.plot()

    # print(classification)
    print("accuracy", accuracy)
    print("precision", precision)
    print("recall", recall)
    print("f1", f1)
    print("kappa", kappa)
    # print('AUC:', auc)

    return matrix, classification, accuracy, precision, recall, f1, kappa



def metric_log2(test_preds, test_score, test_label, args):

    test_label = np.array(test_label)-args.train_labels_minux

    # print("truth: ")
    # data_reader.data_info(np.array(test_label)-1)

    # print("predict: ")
    # data_reader.data_info(np.array(y_pred_test))

    classification = metrics.classification_report(test_label, test_preds, digits=4)
    accuracy = metrics.accuracy_score(test_label, test_preds)
    kappa = metrics.cohen_kappa_score(test_label, test_preds)

    print(classification)
    print("accuracy", accuracy)
    print("kappa", kappa)

    return classification, accuracy, kappa


class Evaluator(object):
    '''
        手动计算分类问题的评价指标

    '''
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.eps = 1e-8
        # self.eps = 0

    def get_tp_fp_tn_fn(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp + self.eps)
        return precision

    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn + self.eps)
        return recall

    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp + self.eps)
        Recall = tp / (tp + fn + self.eps)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall + self.eps)
        return F1

    def OA(self):
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return OA

    def Intersection_over_Union(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp + self.eps)
        return IoU
    
    def m_Intersection_over_Union(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp + self.eps)
        mIoU = 0.5 * (IoU + (tn / (tn + fp + fn + self.eps)))
        return mIoU
    
    def Dice(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / ((tp + fp) + (tp + fn))
        return Dice

    def Pixel_Accuracy_Class(self):
        #         TP                                  TP+FP
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.eps)
        return Acc

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + self.eps)
        iou = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU
    
    def Kappa(self):
        # kappa metric 
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        po = (tp + tn) / (tp + tn + fp + fn)
        pe = ((tp + fp) * (tp + fn) + (fp + tn) * (fn + tn)) / (tp + tn + fp + fn) ** 2
        kappa = (po - pe) / (1 - pe)
        return kappa

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
                                                                                                 gt_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)



if __name__ == '__main__':

    # gt = np.array([[0, 2, 1],
    #                [1, 2, 1],
    #                [1, 0, 1]])

    # pre = np.array([[0, 1, 1],
    #                [2, 0, 1],
    #                [1, 1, 1]])

    # gt = np.array([0, 0, 1, 1, 1, 1, 1, 0])
    # pre = np.array([0, 1, 0, 1, 1, 1, 1, 0])
    gt = np.array([0, 0, 1, 1, 1, 1])
    pre = np.array([0, 1, 0, 1, 1, 1])
    # print(gt.shape, pre.shape)

    eval = Evaluator(num_class=2)
    eval.add_batch(gt, pre)
    # print("confusion_matrix", eval.confusion_matrix)
    print("get_tp_fp_tn_fn", eval.get_tp_fp_tn_fn())
    print("Precision", eval.Precision())
    print("Recall", eval.Recall())
    print("IOU", np.nanmean(eval.Intersection_over_Union()))
    print("mIOU", np.nanmean(eval.m_Intersection_over_Union()))
    print("OA", eval.OA())
    print("F1", eval.F1())
    print("Frequency_Weighted_IOU", eval.Frequency_Weighted_Intersection_over_Union())

