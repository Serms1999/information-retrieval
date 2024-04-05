import matplotlib.pyplot as plt
import numpy as np
from pandas import Series


def generate_relevance_array(relevance_map: dict[set[str], set[str]], categories: Series, relevant_value: int = 1,
                             non_relevant_value: int = -1) -> np.ndarray[list[int]]:
    """
    Generate an array of length `num_queries` containing a list of `relevant_value` for relevant documents and
     `non_relevant_value` for non-relevant, for each query
    :param relevance_map: a dictionary that contains the map between the queries and the categories that are relevant
        for each query
    :param categories: a pandas Series containing the actual categories of the documents
    :return: a numpy array containing the relevances for each query
    """

    num_queries: int = len(relevance_map)
    num_documents: int = len(categories)

    relevances: list[list[int]] = []
    for query in relevance_map:
        query_relevances: list[int] = []
        for document in categories:
            query_relevances.append(relevant_value if document in relevance_map[query] else non_relevant_value)
        relevances.append(query_relevances)

    return np.array(relevances)


def evaluate(method: str, queries: list[np.ndarray], relevances: list[np.ndarray]) -> None:
    """
    Evaluate the performance of an IR system
    :param method: the evaluation method to use
        - `prec_rec`: draw the Precision-Recall curve for each query
        - `r-prec`: Determine the R-precision for each query
        - `map`: Calculate the Mean Average Precision
        - `roc`: Draw the Receiver-Operating-Characteristic for each query
        - `auc`: Compute the Area Under the ROC curve
        - `all`: all of the above
        - `clear`: cleans the solution space
    :param queries: list of queries, each query is a list of documents
    :param relevances: list of relevances, each relevancy is a list of documents, 1 for relevant, -1 for non-relevant
    """

    num_queries = len(queries)
    num_documents = len(queries[0])

    # Convert the relevancies to the interval [0, 1]
    relevances_norm = (np.array(relevances) + 1) / 2

    def compute_precision_recall(verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the precision and recall for each query
        :param verbose: print the results to the console (default: True)
        :return:
        """

        precision_list = []
        recall_list = []

        if verbose:
            print(f'Precision and Recall at k for k=1,...,{num_documents}')

        for query_index in range(num_queries):
            query_relevances = relevances_norm[query_index, :]
            if verbose:
                print(f'\tQuery {query_index + 1}')

            query_precision = []
            query_recall = []

            for index in range(1, num_documents + 1):
                relevant_documents_so_far = np.sum(query_relevances[:index])
                total_relevant_documents = np.sum(query_relevances)

                precision = relevant_documents_so_far / index
                recall = relevant_documents_so_far / total_relevant_documents

                if verbose:
                    print(f'\t\tP({index})={relevant_documents_so_far}/{index}={precision:.2f},\t'
                          f'R({index})={relevant_documents_so_far}/{total_relevant_documents}={recall:.2f}')

                query_precision.append(precision)
                query_recall.append(recall)
            precision_list.append(query_precision)
            recall_list.append(query_recall)

        return np.array(precision_list), np.array(recall_list)

    def compute_true_positive_rate_false_positive_rate(verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the True Positive Rate and False Positive Rate for each query
        :param verbose: print the results to the console (default: True)
        :return:
        """

        tpr_list = []
        fpr_list = []

        if verbose:
            print(f'TP_rate and FP_rate at k for k=1,...,{num_documents}')

        for query_index in range(num_queries):
            query_relevances = relevances_norm[query_index, :]
            query_non_relevant_documents = 1 - relevances_norm[query_index, :]
            if verbose:
                print(f'\tQuery {query_index + 1}')

            query_tpr = []
            query_fpr = []

            number_of_relevant_documents = np.sum(query_relevances)
            number_of_non_relevant_documents = np.sum(query_non_relevant_documents)

            for index in range(1, num_documents + 1):
                tpr_so_far = np.sum(query_relevances[:index]) / number_of_relevant_documents
                fpr_so_far = np.sum(query_non_relevant_documents[:index]) / number_of_non_relevant_documents

                if verbose:
                    print(f'\t\tTP_rate({index})=R({index})={np.sum(query_relevances[:index])}'
                          f'/{number_of_relevant_documents}={tpr_so_far:.2f}\t'
                          f'FP_rate({index})=NR({index})={np.sum(query_non_relevant_documents[:index])}'
                          f'/{number_of_non_relevant_documents}={fpr_so_far:.2f}')

                query_tpr.append(tpr_so_far)
                query_fpr.append(fpr_so_far)
            tpr_list.append(query_tpr)
            fpr_list.append(query_fpr)

        return np.array(tpr_list), np.array(fpr_list)

    if method in ('prec_rec', 'all'):
        total_precision, total_recall = compute_precision_recall()
        print('\n Draw the Precision-Recall curve for each query')

        for query_index in range(num_queries):
            print(f'\tQuery {query_index + 1}')

            plt.figure()
            query_recall = total_recall[query_index, :]
            query_precision = total_precision[query_index, :]

            plt.scatter(query_recall, query_precision)
            plt.plot(query_recall, query_precision, label='Precision-Recall curve')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')

            interpolated_recall = np.hstack([0, query_recall, 1])
            interpolated_precision = np.zeros(query_recall.size)

            for i in range(query_recall.size - 1):
                recall = interpolated_recall[i]
                if i != 0 and interpolated_recall[i + 1] == recall:
                    interpolated_precision[i] = np.max(query_precision[i - 1:])
                else:
                    interpolated_precision[i] = np.max(query_precision[i:])

            plt.plot(interpolated_recall, interpolated_precision, color='red', label='Interpolated PR curve')
            plt.legend(loc='lower left')
            plt.show()

    if method in ('r-prec', 'all'):
        if total_precision is None:
            total_precision, total_recall = compute_precision_recall()
        print('\n Determine R-precision for each query')

        for query_index in range(num_queries):
            query_recall = total_recall[query_index, :]
            query_precision = total_precision[query_index, :]

            relevant_documents = int(np.sum(relevances_norm[query_index]))
            print(f'\tQuery {query_index + 1}')
            print(f'\t\tNumber of relevant documents: {relevant_documents} --> '
                  f'P({relevant_documents})={query_precision[relevant_documents - 1]:.2f}')

    if method in ('roc', 'all', 'auc'):
        TP, FP = compute_true_positive_rate_false_positive_rate()
        print('\n Draw the ROC curve for each query')

        for query_index in range(num_queries):
            print(f'\tQuery {query_index + 1}')

            plt.figure()
            query_tpr = TP[query_index, :]
            query_fpr = FP[query_index, :]

            plt.scatter(query_fpr, query_tpr)
            plt.plot(query_fpr, query_tpr, label='ROC curve')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('FP rate')
            plt.ylabel('TP rate')
            plt.show()

            if method == 'auc' or method == 'all':
                AUC = []
                for i in range(query_tpr.size - 1):
                    delta_x = query_fpr[i + 1] - query_fpr[i]
                    base = query_tpr[i + 1] + query_tpr[i]
                    AUC.append(base * delta_x / 2)
                AUC = np.array(AUC)
                AUC = AUC[AUC > 0]
                string_AUC = ' + '.join([f'{auc:.2f}' for auc in AUC])
                if string_AUC != '':
                    string_AUC += ' = '
                print(f'\tAUC = {string_AUC} {np.sum(AUC):.2f}\n\n')
