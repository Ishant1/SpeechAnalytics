# function to make a confusion matrix given its metadata and filter for the same
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import average_precision_score, PrecisionRecallDisplay
from sklearn.preprocessing import OneHotEncoder
from IPython.core.display import display, HTML


def get_confusion_matrix_with_filter(sids, true_labels, predicted, filter=''):
    """
    Create confusion matrix with true and predicted labels with filters on a id for a utterance
    :param sids: list ids for each row of data used
    :param true_labels: list of true labels
    :param predicted: list of predicted labels
    :param filter: the filter tag to use for filterng results
    :return: a Dataframe with rows as true and columns as predicted labels
    """

    index = [filter in s for s in sids]
    filtered_true = np.array(true_labels)[index]
    filtered_predicted = np.array(predicted)[index]
    return pd.crosstab(filtered_true,filtered_predicted)


def get_confusion_matrix(true_labels, predicted_labels):
    """
    Create confusion matrix with true and predicted labels
    :param true_labels: list of true labels
    :param predicted_labels: list of predicted labels
    :return: a Dataframe with rows as true and columns as predicted labels
    """
    return pd.crosstab(true_labels, predicted_labels)


def get_metrics_with_filter(sids, true_labels, predicted, filter):
    """
    get multiple metrics from true and predicted labels, filtered using sids and filter tag
    :param sids: The id for each result. Can be a list of metadat alike gender, id etc.
    :param true_labels: list of true labels
    :param predicted: list of predicted labels
    :param filter: the filter tag to use for filterng results
    :return: a Dataframe with Precision, Recall, f1-score, Support(no. of labels
    """
    index = [filter in s for s in sids]
    filtered_true = np.array(true_labels)[index]
    filtered_predicted = np.array(predicted)[index]

    return get_metrics(filtered_true,filtered_predicted)


def get_metrics(true_labels, predicted_labels):
    """
    get multiple metrics from true and predicted labels
    :param true_labels: list of true labels
    :param predicted_labels: list of predicted labels
    :return: a Dataframe with Precision, Recall, f1-score, Support(no. of labels
    """
    results_df = pd.DataFrame(np.array(score(true_labels, predicted_labels)).T,
                              columns=['Precision', 'Recall', 'f1-score', 'Support'])
    results_df.index = np.unique(predicted_labels)
    results_df = pd.concat([results_df, pd.DataFrame(
        np.array(score(true_labels, predicted_labels, average='micro')).reshape(1, -1),
        columns=['Precision', 'Recall', 'f1-score', 'Support'], index=['average'])])
    return results_df


def display_side_by_side(dfs:list, captions:list, tablespacing =5, colour = True):
    """
    Display tables side by side to save vertical space, reformat the tables to be colour coded for max and min value per column
    :param dfs: list of pandas.DataFrame
    :param captions: list of table captions
    :param tablespacing: Spacing between tables
    :param colour: Boolean to colour tables min-max per column
    :return: HTML results with dataframe presented side by side
    """

    output = ""
    for (caption, df) in zip(captions, dfs):
      styled_table = df.style.format(precision=2)\
        .set_properties(**{'text-align': 'right'})\
        .set_table_styles([{  # for row hover use <tr> instead of <td>
    'selector': 'td:hover',
    'props': [('background-color', '#ffffb3')]
},
{
    'selector':'td',
 'props':[('padding-left','20px')]
}])\
        .set_table_attributes("style='display:inline'")\
        .set_caption(caption)
      if colour:
        styled_table = styled_table.highlight_max(color='lightgreen').highlight_min(color='#cd4f39')

      output += styled_table._repr_html_()
      output += tablespacing * "\xa0"
    display(HTML(output))


def precision_recall_roc(results, label_to_index, sids=None, filter=''):
    """
    Generates precision-recall and roc metrics for each class. The function uses results json file to get all prediction probs
    :param results: json load of results for a given model. The format of the dict can be seen in utils.results_utils.results_to_json
    :param label_to_index: Dict mapping name of a label and its index in the results
    :param sids: The ids of the results, if given it can be used to filter results
    :param filter: filter tag to filter the results
    :return: precision, recall and average precision as a dict for each class index and micro average. roc_curve_dict: a dict with metrics to create a ROC curve for each class
    """
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    roc_curve_dict = dict()

    ohe = OneHotEncoder()

    labels_index = pd.Series(results['labels']).replace(label_to_index).to_numpy()

    ohe.fit(labels_index.reshape(-1,1))

    if sids:
      index = [filter in s for s in sids]
      filtered_labels = labels_index[index]
      filtered_probs = np.array(results['predicted_probs'])[index,:]

      print(filtered_labels.shape)
      print(filtered_probs.shape)

    else:
      filtered_labels = labels_index
      filtered_probs = np.array(results['predicted_probs'])

    label_matrix = ohe.transform(filtered_labels.reshape(-1,1)).toarray()
    predicted_matrix = filtered_probs
    for i in range(label_matrix.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(label_matrix[:, i], predicted_matrix[:, i])
        average_precision[i] = average_precision_score(label_matrix[:, i], predicted_matrix[:, i])
        roc_curve_dict[i] = roc_curve(label_matrix[:, i], predicted_matrix[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        label_matrix.ravel(), predicted_matrix.ravel()
    )
    average_precision["micro"] = average_precision_score(label_matrix, predicted_matrix, average="micro")

    return precision, recall, average_precision, roc_curve_dict


def get_precision_recall_graphs(recall, precision, average_precision, index_to_label):
    """
    Uses results from precison_recall_roc to create the graphs and title them with respective names
    :param recall: dict of recall metrics for each class for precision-recall curve
    :param precision: dict of precision metrics for each class for precision-recall curve
    :param average_precision: dict of average precision metrics for each class
    :param index_to_label: Dict mapping index of result to name of emotion
    :return: Displays the precision-recall graph for each class and average
    """
    display = PrecisionRecallDisplay(
      recall=recall['micro'],
      precision=precision['micro'],
      average_precision=average_precision['micro'],
  )
    display.plot()
    _ = display.ax_.set_title("Micro Average Precision-Recall")

    for i in index_to_label.keys():
      display = PrecisionRecallDisplay(
          recall=recall[i],
          precision=precision[i],
          average_precision=average_precision[i],
      )
      display.plot()
      _ = display.ax_.set_title(f"{index_to_label[i]}")


def get_roc_graphs(roc_dict,index_to_label):
    """
    Uses results from precison_recall_roc method to create the graphs and title them with respective names
    :param roc_curve_dict: dict of ROC metrics for each class for roc
    :param index_to_label: Dict mapping index of result to name of emotion
    :return: Displays the ROC for each class
    """
    #create ROC curve
    no_of_labels = len(index_to_label.keys())
    fig, ax = plt.subplots(1,no_of_labels,figsize=(5*no_of_labels, 4),sharey='row')
    axs = ax.ravel()
    for i in index_to_label.keys():
        axs[i].plot(roc_dict[i][0],roc_dict[i][1])
        axs[i].set(xlabel='False Positive Rate', ylabel='True Positive Rate',
            title=f"{index_to_label[i]} ROC")

    plt.show()