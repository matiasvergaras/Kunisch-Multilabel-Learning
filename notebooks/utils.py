# Clases utiles para usar en los notebooks.
import numpy as np
import pandas as pd
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, precision_score, recall_score, fbeta_score
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import sparse
import random


class KunischMetrics:
    """
    KunischMetrics: clase para generar métricas en base a una matriz de etiquetas ground_truth y una de predicciones
    """

    def __init__(self, y_true, y_pred):
        """
        :param y_true: ground truth
        :param y_pred: predicciones a evaluar
        """
        assert y_true.shape == y_pred.shape
        if sparse.issparse(y_true):
            y_true = y_true.todense()
        if sparse.issparse(y_pred):
            y_pred = y_pred.todense()
        self.y_true = y_true
        self.y_pred = y_pred

    def f1(self, average='micro'):
        """
        F1-Score
        :param average: estrategia micro/macro/weighted
        :return: F1-Score calculado según 'average' con 4 decimales de precisión.
        """
        return round(f1_score(self.y_true, self.y_pred, average=average), 4)

    def f2(self, beta=2, average='micro'):
        """
        F2-Score
        :param average: estrategia micro/macro/weighted
        :return: F2-Score calculado según 'average' con 4 decimales de precisión.
        """
        return round(fbeta_score(self.y_true, self.y_pred, beta=beta, average=average), 4)

    def recall(self, average='micro'):
        """
        Recall
        :param average: estrategia micro/macro/weighted
        :return: Recall calculado según 'average' con 4 decimales de precisión.
        """
        return round(recall_score(self.y_true, self.y_pred, average=average), 4)

    def precision(self, average='micro'):
        """
        Precision
        :param average: estrategia micro/macro/weighted
        :return: Precision calculado según 'average' con 4 decimales de precisión.
        """
        return round(precision_score(self.y_true, self.y_pred, average=average), 4)

    def acc(self, normalize=True):
        """
        Accuracy
        :param normalize: boolean
        :return: Accuracy con 4 decimales de precisión, normalizado o no según normalize.
        """
        return round(accuracy_score(self.y_true, self.y_pred, normalize=normalize), 4)

    def hl(self):
        """
        Hamming Loss: fracción de etiquetas que son incorrectamente predichas.
        :return: Hamming Loss con 4 decimales de precisión.
        """
        return round(hamming_loss(self.y_true, self.y_pred), 4)

    def emr(self):
        """
        Exact Match Ratio
        :return: Exact Match Ratio con 4 decimales de precisión.
        """
        return round(np.all(self.y_pred == self.y_true, axis=1).mean(), 4)

        # Compute Hamming Score


    def hs(self):
        """
        Hamming Score
        Hamming Score = |Intersección de positivos|/|Unión de positivos|, promediado por la cantidad de samples
        También se puede ver como la proporción de etiquetas correctamente asignadas sobre la cantidad total de
        etiquetas asignadas. Se conoce además como Multilabel Accuracy, y "castiga" por: (1) no predecir una etiqueta
        correcta (disminuyendo la cardinalidad de la intersección) y (2) incluir una etiqueta incorrecta (aumentando
        la cardinalidad de la unión).
        :return: Hamming Score con 4 decimales de precisión.
        """
        acc_list = []
        for i in range(self.y_true.shape[0]):
            set_true = set(np.where(self.y_true[i])[0])
            set_pred = set(np.where(self.y_pred[i])[0])
            if len(set_true) == 0 and len(set_pred) == 0:
                tmp_a = 1
            else:
                tmp_a = len(set_true.intersection(set_pred)) / \
                        float(len(set_true.union(set_pred)))
            acc_list.append(tmp_a)
        return round(np.mean(acc_list), 4)

    # Nos gustaria tener una metrica que mida al menos cuantas etiquetas son predichas correctamente para cada patron
    # Y quizá sería interesante relacionar eso con el label cardinality (que en nuestro dataset es 5.28)
    def k_match_ratio(self, n=5):
        """
        K-Match Ratio
        Proporción de patrones con al menos K etiquetas correctamente asignadas
        :param n: parámetro K de K-Match Ratio
        :return: K Match Ratio con 4 decimales de precisión.
        """
        count = 0
        for i in range(self.y_true.shape[0]):
            tp = np.logical_and(self.y_true[i], self.y_pred[i])
            tp = np.sum(tp)
            eq = np.sum(self.y_true[i]) == tp
            # print(eq, np.sum(self.y_true[i]), tp)
            count += 1 if (tp >= n or eq) else 0
        return round(count / self.y_true.shape[0], 4)

    def mr1(self):
        """
        :return: 1-Match Ratio
        """
        return self.k_match_ratio(1)

    def mr2(self):
        """
        :return: 2-Match Ratio
        """
        return self.k_match_ratio(2)

    def mr3(self):
        """
        :return: 3-Match Ratio
        """
        return self.k_match_ratio(3)

    def mr4(self):
        """
        :return: 4-Match Ratio
        """
        return self.k_match_ratio(4)

    def mr5(self):
        """
        :return: 5-Match Ratio
        """
        return self.k_match_ratio(5)


class KunischPruner:
    """
    KunischPruner: clase para 'recortar' un archivo de etiquetas para dejar solamente las solicitadas en un arreglo
    de desired_labels
    """

    def __init__(self, desired_labels):
        self.desired_labels = desired_labels
        self.final_labels = -1
        self.top_labels = None

    def set_top_labels(self, top_labels):
        """
        Setea la variable top_labels con un dataframe recibido por parámetro.
        :param top_labels: dataframe de top labels.
        :return:
        """
        self.top_labels = top_labels

    def filter_labels(self, labels_df, pruning_freq=None):
        """
        Recibe un dataframe de labels y lo filtra según pruning_freq (si se ingresa dicho valor)
        o según self.desired_labels en caso contrario.
        La primera opción filtra según frecuencia (dejando todas las etiquetas con una frecuencia
        mayor o igual a la solicitada). La segunda opción filtra según cantidad de etiquetas a
        mantener.
        :param labels_df: dataframe de etiquetas
        :param pruning_freq: threshold de frecuencia
        :return: dataframe de etiquetas más frecuentes
        """
        top_labels = None

        if pruning_freq is not None:
            print(f"Utilizando pruning frequency.\
            Se ignorará la cantidad deseada de etiquetas para cortar en {pruning_freq} eventos.")
            filtered_df = labels_df.loc[:, labels_df.sum(axis=0) > pruning_freq]
            top_labels = filtered_df.sum().sort_values(ascending=False)
            return top_labels

        else:
            filtered_labels = labels_df.shape[1]
            pivot = 0
            while filtered_labels > self.desired_labels:
                filtered_df = labels_df.loc[:, labels_df.sum(axis=0) > pivot]
                top_labels = filtered_df.sum().sort_values(ascending=False)
                filtered_labels = filtered_df.shape[1]
                pivot += 1
            print("Aplicando threshold {} para trabajar con {} labels".format(pivot, len(top_labels.values)))
            self.final_labels = len(top_labels.values)
            self.top_labels = top_labels.values
            return top_labels

    def filter_df(self, df):
        """
        Recibe un dataframe y lo filtra según self.top_labels, retornando
        solamente las columnas en común.
        :param df: dataframe a filtrar
        :return: dataframe intersectado con self.top_labels
        """
        df = df[df.columns.intersection(self.top_labels.index)]
        return df


#
class DataExplorer:
    """
     DataExplorer: clase para reportar algunas métricas de exploración de datos interesantes, tales como
        el Label Density o el Label Average.
    """

    def __init__(self, train_labels, val_labels, test_labels):
        """
        Un DataExplorer comienza recibiendo los df de train, val y test.
        :param train_labels: df con etiquetas de entrenamiento
        :param val_labels: df con etiquetas de validación
        :param test_labels: df con etiquetas de test
        """
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels

    def get_unique_combinations(self, study='train'):
        """
        Entrega la cantidad de combinaciones únicas de etiquetas en un df especificada por
        el parámetro study
        :param study: 'train', 'val', 'test' o 'all'. Especifica el df a estudiar.
        :return: Cantidad de combinaciones únicas de etiquetas en el df solicitado.
        """
        if study == 'train':
            labels_df = self.train_labels
        elif study == 'val':
            labels_df = self.val_labels
        elif study == 'test':
            labels_df = self.test_labels
        elif study == 'all':
            labels_df = pd.concat([self.train_labels, self.val_labels, self.test_labels])

        unique_combinations = len(labels_df.drop_duplicates())
        print(f"Number of unique labels combinations in {study}: {unique_combinations}")
        return unique_combinations

    def get_label_metrics(self, study='train'):
        """
        Entrega label density y label average de un df especificado por el parámetro study.
        :param study: 'train', 'val', 'test' o 'all'. Especifica el df a estudiar.
        :return: LD y LA del df solicitado.
        """
        if study == 'train':
            labels_df = self.train_labels
        elif study == 'val':
            labels_df = self.val_labels
        elif study == 'test':
            labels_df = self.test_labels
        elif study == 'all':
            labels_df = pd.concat([self.train_labels, self.val_labels, self.test_labels])

        sum_labels = labels_df.sum(axis=1)
        total_labels = labels_df.shape[0]
        total_patterns = labels_df.shape[1]
        label_cardinality = 0
        for label in sum_labels:
            label_cardinality += label / total_labels
        label_density = label_cardinality / total_patterns
        print("Label cardinality in {}: {}".format(study, label_cardinality))
        print("Label density in {}: {}".format(study, label_density))
        return label_cardinality, label_density


class KunischPlotter:
    """
    KunischPlotter: clase para crear visualizaciones de los resultados.
    """
    def __init__(self, num_lines=30):
        """
        Un KunischPlotter comienza creando una serie de linemarks a utilizar en los
        gráficos a crear.
        :param num_lines: cantidad de linemarks a crear, 30 por defecto.
        """
        linemarks = []
        # MARKERS = ['.', '+', 'v', 'x', '*']
        MARKERS = ['+']
        LINE_STYLES = ['--']  # , '--', '-.', ':']

        for i in range(0, num_lines):
            linestyle = LINE_STYLES[random.randint(0, len(LINE_STYLES) - 1)]
            marker = MARKERS[random.randint(0, len(MARKERS) - 1)]
            # color = COLORCYCLE[i % len(COLORCYCLE)]
            linemarks.append(linestyle + marker)

        self.linemarks = linemarks

    def plot_results(self, x, score=[], label=[], title="", xlabel="", ylabel="", width=7, height=9,
                     y_low_lim=0.0,
                     ylim=0.6, xlim=300,
                     order=None, grid=False, minorgrid=False, markersize=4,
                     colors=None):
        """
        Plotter de gráfico de curvas con el desempeño de los métodos.
        Los parámetros siguen la misma nomenclatura de matplotlib.
        """
        assert len(x) == len(score[0])
        fig = plt.figure(1)
        fig.set_figheight(height)
        fig.set_figwidth(width)
        ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(y_low_lim, ylim)
        ax.set_xlim(0, xlim)

        cm = sns.color_palette('tab20', n_colors=len(score))

        for i in range(0, len(score)):
            line = ax.plot(x, score[i], self.linemarks[i % len(self.linemarks)], label=label[i], markersize=markersize)
            if colors is not None:
                line[0].set_color(colors[i])
            else:
                line[0].set_color(cm[i % len(cm)])
        ax.legend()

        if grid:
            ax.grid(which='both')

        if minorgrid:
            ax.minorticks_on()
            # Customize the major grid
            ax.grid(which='major', linestyle='-', linewidth='0.5')
            # Customize the minor grid
            ax.grid(which='minor', linestyle=':', linewidth='0.5')

        # labels en orden deseado
        if order is not None:
            handles, labels = plt.gca().get_legend_handles_labels()
            order = order
            ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

        fig.show()

    def print_confusion_matrix(self, cm, axes, class_label, class_names, fontsize=14, normalize=True):
        """
        Plotter de matriz de confusión
        Los parámetros siguen la misma nomenclatura de matplotlib.
        """
        df_cm = pd.DataFrame(
            cm, index=class_names, columns=class_names,
        )
        if normalize:
            # print(df_cm)
            df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
            # print(df_cm)
        heatmap = sns.heatmap(df_cm, annot=True, fmt=".2f", cbar=False, ax=axes, cmap='Blues',
                              annot_kws={"size": fontsize})
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        axes.set_ylabel('True label')
        axes.set_xlabel('Predicted label')
        axes.set_title(class_label)

    def plot_multiple_matrix(self, cfs_matrix, present_labels, nrows=5, ncols=5, figsize=(6, 10), filename="cm",
                             normalize=True, fontsize=14):
        """
        Plotter de múltiples matrices de confusión
        Los parámetros siguen la misma nomenclatura de matplotlib.
        """
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

        for axes, cfs_vals, label in zip(ax.flatten(), cfs_matrix, present_labels):
            self.print_confusion_matrix(cfs_vals, axes, label, ["N", "Y"], normalize=normalize, fontsize=fontsize)

        fig.tight_layout()
        plt.show()
