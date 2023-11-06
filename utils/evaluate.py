from sklearn import metrics
import numpy as np
from collections import defaultdict


def metric(y_true, y_pred, y_prob, empty=-1, multilabel=False):
    assert len(y_true) == len(y_pred) == len(y_prob)

    if not multilabel:
        y_true, y_pred, y_prob = np.array(y_true).flatten(), np.array(y_pred).flatten(), np.array(y_prob).flatten()
        # filter data
        flag = y_true != empty
        y_true, y_pred, y_prob = y_true[flag], y_pred[flag], y_prob[flag]
    else:
        y_true, y_pred, y_prob = np.array(y_true).flatten(), np.array(y_pred).flatten(), np.array(y_prob)

    if not multilabel:
        auc = metrics.roc_auc_score(y_true, y_prob)
        return {"ROCAUC": auc}
    else:
        auc = metrics.roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        return {"ROCAUC": auc}


def metric_reg(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()

    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }


def metric_multitask(y_true, y_pred, y_prob, num_tasks, empty=-1):
    assert num_tasks == y_true.shape[1] == y_pred.shape[1] == y_prob.shape[1]
    assert y_prob.min() >= 0 and y_prob.max() <= 1

    result_list_dict_each_task = []

    cur_num_tasks = 0
    for i in range(num_tasks):
        flag = y_true[:, i] != empty
        if len(y_true[flag, i].flatten()) == 0 or len(set(y_true[flag, i].flatten())) == 1:  # data is none or labels are all one value
            result_list_dict_each_task.append(None)
        else:
            result_list_dict_each_task.append(metric(y_true[flag, i].flatten(), y_pred[flag, i].flatten(), y_prob[flag, i].flatten()))
            cur_num_tasks += 1

    mean_performance = defaultdict(float)
    for i in range(num_tasks):
        if result_list_dict_each_task[i] is None:
            continue
        for key in result_list_dict_each_task[i].keys():
            if key == "fpr" or key == "tpr" or key == "precision_list" or key == "recall_list":
                continue
            mean_performance[key] += result_list_dict_each_task[i][key] / cur_num_tasks

    mean_performance["result_list_dict_each_task"] = result_list_dict_each_task

    if cur_num_tasks < num_tasks:
        print("Some target is missing! Missing ratio: {:.2f} [{}/{}]".format(1 - float(cur_num_tasks) / num_tasks,
                                                                             cur_num_tasks, num_tasks))
        mean_performance["some_target_missing"] = "{:.2f} [{}/{}]".format(1 - float(cur_num_tasks) / num_tasks,
                                                                          cur_num_tasks, num_tasks)

    return mean_performance


def metric_reg_multitask(y_true, y_pred, num_tasks):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    assert num_tasks == y_true.shape[1] == y_pred.shape[1]
    assert len(y_true) == len(y_pred)

    result_list_dict_each_task = []
    for i in range(num_tasks):
        result_list_dict_each_task.append(metric_reg(y_true[:, i].flatten(), y_pred[:, i].flatten()))

    mean_performance = defaultdict(float)
    for i in range(num_tasks):
        for key in result_list_dict_each_task[i].keys():
            mean_performance[key] += result_list_dict_each_task[i][key] / num_tasks

    mean_performance["result_list_dict_each_task"] = result_list_dict_each_task

    return mean_performance


def calculate_multitask_roc_auc(fprs, tprs):
    assert len(fprs) == len(tprs)
    num_tasks = len(fprs)

    all_fpr = np.unique(np.concatenate([fprs[i] for i in range(num_tasks)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_tasks):
        mean_tpr += np.interp(all_fpr, fprs[i], tprs[i])
    mean_tpr /= num_tasks
    macro_fpr = all_fpr
    macro_tpr = mean_tpr
    macro_roc_auc = metrics.auc(macro_fpr, macro_tpr)

    return macro_roc_auc, macro_fpr, macro_tpr

