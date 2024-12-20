import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


labels_for_plots = {
    "MemoryComplaints": ["No", "Yes"],
    "BehavioralProblems": ["No", "Yes"],
}


def kde_and_boxplots(
    data: pd.DataFrame, column: str, target: str
) -> matplotlib.figure.Figure:
    """
    Отрисовка boxplot и kdeplot для числовых признаков
    :param data: датасет
    :param column: признак, анализируемый в разрезе целевой переменной
    :param target: целевая переменная
    :return: поле рисунка
    """
    fig, axes = plt.subplots(ncols=2, figsize=(11, 6))
    sns.kdeplot(
        data, x=column, hue=target, palette="rocket", common_norm=False, ax=axes[0]
    )

    sns.boxplot(
        data, y=column, x=target, legend=False, palette="rocket", hue=target, ax=axes[1]
    )
    plt.suptitle(f"{column} - Diagnosis", fontsize=15)

    return fig


def barplot_norm_target(
    data: pd.DataFrame, column: str, target: str, labels: dict = labels_for_plots
) -> matplotlib.figure.Figure:
    """
    Построение barplot с нормированными данными с выводом значений на графике
    """
    norm_target = (
        data.groupby(target)[column]
        .value_counts(normalize=True)
        .mul(100)
        .rename("percent")
        .reset_index()
    )
    fig = plt.figure(figsize=(7, 6))
    ax = sns.barplot(
        data=norm_target, x=column, y="percent", hue=target, palette="crest"
    )
    plt.xticks(ticks=range(len(labels[column])), labels=labels[column])
    plt.ylim((0, 100))
    for p in ax.patches:
        percentage = f"{p.get_height():.1f}%"
        ax.annotate(
            percentage,  # текст
            # координата xy
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            # центрирование
            ha="center",
            va="center",
            xytext=(0, 10),
            # точка смещения относительно координаты
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel(column, fontsize=9)
    plt.ylabel("Процент", fontsize=9)
    plt.title(f"{column} - Diagnosis", fontsize=10)

    return fig


def plot_feature_importances(perm_path: str) -> matplotlib.figure.Figure:
    """
    Отрисовка permutation importances в виде barplot
    :param perm_path: путь до .csv файла с permutation importances
    """
    fig = plt.figure()
    perm_df = pd.read_csv(perm_path)
    sns.barplot(data=perm_df[:15], x="value", y="feature", palette="viridis")
    plt.title("Значимость признаков (Permutation importance)")

    return fig
