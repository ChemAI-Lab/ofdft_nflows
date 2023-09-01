import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


def latex_format(input_string):
    return_string = input_string
    if input_string.isnumeric():
        return_string = f"{return_string}_{{{input_string}}}"
    elif input_string[-1].isdigit():
        # Find the last digit in the input string
        last_digit_index = len(input_string) - 1
        while last_digit_index >= 0 and input_string[last_digit_index].isdigit():
            last_digit_index -= 1

        # Insert "_{" and "}" after the last digit
        return_string = (
            input_string[:last_digit_index + 1] +
            f"_{{{input_string[last_digit_index + 1:]}}}"
        )

    return "$" + return_string + "$"


def plot_training_trajectory(rwd: str, mol_name: str, bool_save: bool = False):

    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS)

    mol_name_latex = latex_format(mol_name)

    n_init = 10  # remove some of the noisy initial points

    file = rwd + f'training_trajectory_{mol_name}_ema.csv'
    results_ema = pd.read_csv(file)

    file = rwd + f'training_trajectory_{mol_name}.csv'
    results = pd.read_csv(file)

    keys = results.columns
    epochs = results['epoch'].to_numpy()

    labels_keys = {'E': r"$E[\rho(x)]$", 'T': r"$T_{TFDW}[\rho(x)]$", 'V': r"$V_{e-N}[\rho(x)]$",
                   'H': r"$V_{H}[\rho(x)]$", 'X': r"$V_{X}[\rho(x)]$"}

    fig, ax = plt.subplots()
    for i, k in enumerate(keys[1:-1]):
        r_ = results[k].to_numpy()
        r_ema = results_ema[k].to_numpy()

        ax.plot(epochs[n_init:], r_[n_init:], ls='--', c=colors[i], lw=.2)
        ax.plot(epochs[n_init:], r_ema[n_init:], c=colors[i],
                zorder=1.5, lw=2., label=labels_keys[k])
        # ax.text(0.75*epochs[-1],r_ema[-1]*(1.1),labels_keys[k],)

    ax.text(0.075, 0.95,
            mol_name_latex, transform=ax.transAxes, va='top', fontsize=18)
    plt.ylabel('Funcitonal value [a.u.]', fontsize=15)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylim(top=1.3*np.max(results_ema['T'].to_numpy()[n_init:]))
    plt.legend(loc=1)
    plt.tight_layout()

    if bool_save:
        plt.savefig(f'fig_epochs_vs_values_{mol_name}.svg', transparent=True)
        plt.savefig(f'fig_epochs_vs_values_{mol_name}.png')
    else:
        plt.show()


if __name__ == "__main__":
    rwd = '/Users/ravh011/Documents/GitHub/ofdft_normflows/Results_SCRATCH/H2_TF-W_V_H_X_lr_3.0e-04_sched_MIX/'

    mol_name = 'H2'
    plot_training_trajectory(rwd, mol_name, True)

    rwd = '/Users/ravh011/Documents/GitHub/ofdft_normflows/Results_SCRATCH/H_TF-W_V_H_X_lr_3.0e-04/'

    mol_name = 'H'
    plot_training_trajectory(rwd, mol_name, True)
