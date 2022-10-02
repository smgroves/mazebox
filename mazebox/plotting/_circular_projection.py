from itertools import permutations
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# for a specific set of endpoints:
# arrange endpoints evenly around a circle (i.e. with angles a_k) for each k
# calculare x_i and y_i for each datapoint
# plot in cartesian coordinates
def proximity(probabilities):
    # for each pair of arcs, calculate proximity as sum of cross product
    # list of pairs in order
    l = [[i, i + 1] for i in range(len(probabilities.columns) - 1)]
    l.append([len(probabilities.columns) - 1, 0])
    best_dlk = 0
    best_perm = None
    for perm in list(permutations(probabilities.columns, len(probabilities.columns))):
        d_lk = 0
        for pair in l:
            # print(np.dot(probabilities[perm[pair[0]]], probabilities[perm[pair[1]]]))
            d_lk += np.dot(probabilities[perm[pair[0]]], probabilities[perm[pair[1]]])
        if d_lk > best_dlk:
            best_dlk = d_lk
            best_perm = perm
    return best_dlk, best_perm


def cap_plot(
    probabilities,
    hue=None,
    plot_type="scatterplot",
    fill=False,
    levels=10,
    arc_hue_dict={
        "SCLC-A_Score_pos": "r",
        "SCLC-P_Score_pos": "blue",
        "SCLC-A2_Score_pos": "orange",
        "SCLC-N_Score_pos": "green",
        "SCLC-Y_Score_pos": "purple",
    },
    save=None,
    legend=True,
    set_style="white",
):

    probabilities = probabilities.dropna()

    n_arcs = len(probabilities.columns)

    best_dlk, best_perm = proximity(probabilities)
    # single assignment

    angles = np.linspace(0, 360, n_arcs, endpoint=False)
    # assign arcs to angles
    print(best_perm)
    ang_dict = {i: k for i, k in zip(best_perm, angles)}
    # given assignment, project probabilities to cartesian coordinates
    # each data point
    # x_i
    x = [
        probabilities[list(best_perm)]
        .loc[i]
        .dot(np.cos(np.radians([ang_dict[o] for o in best_perm])))
        for i, r in probabilities.iterrows()
    ]
    # y_i
    y = [
        probabilities[list(best_perm)]
        .loc[i]
        .dot(np.sin(np.radians([ang_dict[o] for o in best_perm])))
        for i, r in probabilities.iterrows()
    ]

    # archetype locations
    x_arc = np.cos(np.radians([ang_dict[o] for o in best_perm]))
    y_arc = np.sin(np.radians([ang_dict[o] for o in best_perm]))
    sns.set_style(set_style)
    plt.figure(figsize=(10, 10))
    if plot_type == "scatterplot":
        sns.scatterplot(
            x, y, s=60, alpha=0.5, hue=hue.loc[probabilities.index], legend=legend
        )
        sns.scatterplot(x_arc, y_arc, s=100, c=[arc_hue_dict[i] for i in best_perm])
    elif plot_type == "kde":
        sns.scatterplot(x_arc, y_arc, s=100, c=[arc_hue_dict[i] for i in best_perm])
        sns.kdeplot(
            x,
            y,
            hue=hue.loc[probabilities.index],
            alpha=0.5,
            fill=fill,
            levels=levels,
            legend=legend,
        )
    if type(save) != type(None):
        plt.savefig(f"./figures/{save}")
    plt.show()
    return best_perm
