import matplotlib.pyplot as plt
import numpy as np
import math


class DataPlots:

    def __init__(self):
        self.fig_width = 20
        self.fig_height = 10

    def pie_plot(self, labels, weights, tag1, tag2):
        fig, ax = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))
        labels_count = [str(labels[i]) + ' (' + str(weights[i]) + ')' for i in range(len(weights))]
        labels_pct = [str(labels[i]) + ' (' + str(round(100 * weights[i] / sum(weights), 1)) + '%)' for i in
                      range(len(weights))]
        explode = [0.1] * len(weights)
        ax.pie(x=weights, explode=explode, labels=labels_count, autopct='%1.1f%%',
               shadow=True, textprops={'fontsize': 12})
        ax.set_title(f'REVIEWS GROUPED BY {tag1.upper()} ({tag2.lower()} total reviews = {sum(weights)})',
                     fontweight='bold', fontsize=18)
        ax.legend(labels=labels_pct, loc='upper left', bbox_to_anchor=(1.1, 1), fancybox=True, shadow=True)
        ax.grid(visible=True)
        fig.tight_layout()
        plt.savefig(f'Reviews grouped by {tag1.lower()} {tag2.lower()}.png', bbox_inches='tight')
        plt.close()

    def plot_most_common_words(self, ratings, repeat, repeat_ind, features, max_words_original, tag):
        ncolumns = 2
        naxis = len(ratings)
        fig, axes = plt.subplots(math.ceil(naxis / ncolumns), ncolumns, figsize=(self.fig_width, self.fig_height))
        spare_axes = ncolumns - naxis % ncolumns
        if spare_axes == ncolumns:
            spare_axes = 0
        for axis in range(ncolumns - 1, ncolumns - 1 - spare_axes, -1):
            fig.delaxes(axes[math.ceil(naxis / ncolumns) - 1, axis])
        ax = axes.ravel()
        for i in range(len(ratings)):
            max_words = min(max_words_original, len(features[i]))
            xtick = features[i][repeat_ind[i][:max_words]]
            top_words = repeat[i][repeat_ind[i][:max_words]]
            ax[i].bar(range(1, max_words + 1), top_words, color='b', width=0.25, edgecolor='black')
            ax[i].set_xticks(range(1, max_words + 1), xtick, ha='center', rotation=75, fontsize=12)
            ax[i].set_title(f'{tag} words in reviews with rating {ratings[i]}', fontsize=24, fontweight='bold')
            ax[i].set_xlabel('Words (Total words = ' + str(len(repeat[i])) + ')', fontweight='bold', fontsize=14)
            ax[i].set_ylabel('Occurrences', fontweight='bold', fontsize=14)
            ax[i].grid(visible=True)
        fig.tight_layout()
        plt.savefig(f'{tag} words per rating.png', bbox_inches='tight')
        plt.close()

    def plot_shared_words(self, ratings, matrix):
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        plt.pcolormesh(matrix, cmap=plt.cm.PuBuGn)
        plt.colorbar()
        ax.set_title('Common words shared between different ratings', fontsize=24, fontweight='bold')
        ax.set_xticks(np.arange(0.5, len(ratings) + 0.5), ratings, fontsize=14, fontweight='bold')
        ax.set_yticks(np.arange(0.5, len(ratings) + 0.5), ratings, fontsize=14, fontweight='bold')
        for i in range(len(ratings)):
            for j in range(len(ratings)):
                ax.text(i + 0.5, j + 0.5, str(matrix[i, j]),
                        ha="center", va="center", color="k", fontweight='bold', fontsize=10)
        fig.tight_layout()
        plt.savefig('Word sharing.png', bbox_inches='tight')
        plt.close()

    def plot_lengths(self, ratings, lengths):
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        ax.bar(ratings, lengths, color='b', width=0.25, edgecolor='black')
        ax.set_title(f'Review length grouped by rating', fontsize=24, fontweight='bold')
        ax.set_xlabel('Rating', fontweight='bold', fontsize=14)
        ax.set_ylabel('Mean length review', fontweight='bold', fontsize=14)
        ax.grid(visible=True)
        for i in range(len(ratings)):
            ax.text(i+1, lengths[i]+5, round(lengths[i], 1), ha="center", va="center", color="k",
                    fontweight='bold', fontsize=10)
        fig.tight_layout()
        plt.savefig(f'Review length.png', bbox_inches='tight')
        plt.close()

    def plot_length_histogram(self, ratings, lengths):
        fig, axes = plt.subplots(3, 2, figsize=(self.fig_width, self.fig_height))
        fig.delaxes(axes[2, 1])
        ax = axes.ravel()
        for i in range(len(ratings)):
            ax[i].hist(lengths[i], histtype='stepfilled', bins=25, alpha=0.25, color="#0000FF", lw=0)
            ax[i].tick_params(axis='both', labelsize=8)
            ax[i].set_title(f'Review lengths with rating {ratings[i]} (total reviews = {len(lengths[i])})',
                            fontsize=24, fontweight='bold')
            ax[i].set_xlim(0, max(max(lengths)))
            ax[i].set_xlabel('Review length', fontweight='bold', fontsize=14)
            ax[i].set_ylabel('Occurrences', fontweight='bold', fontsize=14)
            ax[i].grid(visible=True)
        fig.tight_layout()
        plt.savefig(f'Review lengths per rating.png', bbox_inches='tight')
        plt.close()

    def plot_results(self, tag, test, acc, val_acc, loss, val_loss):
        fig, axes = plt.subplots(2, 1, figsize=(self.fig_width, self.fig_height))
        ax = axes.ravel()
        ax[0].plot(range(1, len(acc) + 1), acc, ls='-', lw=2, color='b', label='Training Accuracy')
        ax[0].plot(range(1, len(val_acc) + 1), val_acc, ls='-', lw=2, color='r', label='Validation Accuracy')
        ax[0].set_xlabel('Epochs', fontweight='bold', fontsize=14)
        ax[0].set_ylabel('Accuracy', fontweight='bold', fontsize=14)
        ax[1].plot(range(1, len(loss) + 1), loss, ls='--', lw=2, color='b', label='Training Loss')
        ax[1].plot(range(1, len(val_loss) + 1), val_loss, ls='--', lw=2, color='r', label='Validation Loss')
        ax[1].set_xlabel('Epochs', fontweight='bold', fontsize=14)
        ax[1].set_ylabel('Loss', fontweight='bold', fontsize=14)
        ax[1].set_ylim(-0.1, val_loss[-1] * 2)
        tstr = ''
        tstr += '\nTRAINING: accuracy = ' + str(round(max(acc), 4)) + ' & loss = ' \
                + str(round(loss[acc.index(max(acc))], 4))
        tstr += '\nVALIDATION: accuracy = ' + str(round(max(val_acc), 4)) + ' & loss = ' \
                + str(round(val_loss[val_acc.index(max(val_acc))], 4))
        tstr += '\nTEST: accuracy = ' + str(round(test[1], 4)) + ' & loss = ' + str(round(test[0], 4))
        fig.suptitle('RESULTS ' + tag.upper() + tstr, fontweight='bold', fontsize=20)
        ax[0].legend()
        ax[1].legend()
        ax[0].grid(visible=True)
        ax[1].grid(visible=True)
        fig.tight_layout()
        plt.savefig('Results ' + tag + '.png', bbox_inches='tight')
        plt.close()

    def linearmodels_coeffs_plot(self, tag, max_feats, max_coeffs, min_feats, min_coeffs):
        fig, axes = plt.subplots(1, 2, figsize=(self.fig_width, self.fig_height))
        ax = axes.ravel()
        ax[0].barh(range(1, len(max_coeffs) + 1), max_coeffs, color='b', height=0.25, edgecolor='black')
        ax[0].set_yticks(range(1, len(max_feats) + 1), max_feats, va='center', rotation=0)
        ax[1].barh(range(1, len(min_coeffs) + 1), min_coeffs, color='r', height=0.25, edgecolor='black')
        ax[1].set_yticks(range(1, len(min_feats) + 1), min_feats, va='center', rotation=0)
        for i in range(2):
            ax[i].grid(visible=True)
            ax[i].set_xlabel('Coefficients', fontsize=14)
            ax[i].set_ylabel('Feature names', fontsize=14)
        fig.suptitle(tag + ' coefficient analysis', fontweight='bold', fontsize=24)
        fig.tight_layout()
        plt.savefig(tag + ' coefficient analysis.png', bbox_inches='tight')
        plt.close()

