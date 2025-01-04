import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import numpy as np

def plot(x, colors):
    # Choosing color palette
    # https://seaborn.pydata.org/generated/seaborn.color_palette.html
    palette = np.array(sns.color_palette("pastel", 10))
    # pastel, husl, and so on

    # Create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int8)])
    # Add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)
    plt.show()

def plot2(data, x='x', y='y'):
    sns.set_context("notebook", font_scale=1.1)
    sns.set_style("ticks")
    # sns.set(style='whitegrid')
    sns.set_palette('bright')
    sns.lmplot(x=x,
            y=y,
            data=data,
            fit_reg=False,
            legend=True,
            height=9,
            hue='Label')

    plt.title('t-SNE Results with Different Triggers', weight='bold').set_fontsize('25')
    # plt.xlabel(x, weight='bold').set_fontsize('20')
    # plt.ylabel(y, weight='bold').set_fontsize('20')
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])

    ax = plt.gca()
    # ax.spines['bottom'].set_color('black')
    # ax.spines['top'].set_color('black')
    # ax.spines['left'].set_color('black')
    # ax.spines['right'].set_color('black')
    plt.subplots_adjust(top=0.9)
    plt.show()
