import matplotlib
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v

def Plot_Results():
    Eval_all = np.load('Journ_Eval_all.npy', allow_pickle=True)
    Terms = ['NSE', 'R', 'RRMASE', 'WI']
    # Terms = []
    Algorithm = ['SSA', 'DHOA', 'EHO', 'PFA', 'PROPOSED']
    Classifier = ['SVR', 'MPNN', 'EL', 'BiLSTM', 'RNN', 'ENSEMBLE', 'PROPOSED']
    for u in range(len(Eval_all)):


        Acc_Table = np.zeros((len(Algorithm)+len(Classifier), len(Terms)))
        for j in range(len(Algorithm) + len(Classifier)):
            for k in range(len(Terms)):
                Acc_Table[j, k] = Eval_all[u, 3, j, k]
        Table = PrettyTable()
        Table.add_column('TERMS', Terms[0:])
        for k in range(len(Algorithm)):
            Table.add_column(Algorithm[k], Acc_Table[k, :])
        print('-------------------------------------------------- Dataset-', u + 1, '-',
              'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)
        print()

        Table = PrettyTable()
        Table.add_column('TERMS', Terms[0:])
        for k in range(len(Classifier)):
            tab = Acc_Table[k+5, :]
            Table.add_column(Classifier[k], tab)
        print('-------------------------------------------------- Dataset-', u + 1, '-',
              'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)
        print()


def plot_results_Statistical():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'R-0.15', 'R-0.25', 'R-0.38', 'R-0.45', 'R-0.55']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(3):
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = stats(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Dataset - ', i + 1, ' Statistical Report ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(25)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='R-0.15')
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='R-0.25')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='R-0.38')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='R-0.45')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='R-0.55')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Correction_Results/Conv_%s.png" % (i + 1))
        plt.show()


def Journ_Plot_Results():
    Eval_all = np.load('Journ_Eval_all.npy', allow_pickle=True)
    Terms = ['NSE', 'R', 'RRMASE', 'WI']
    # Terms = []
    Algorithm = ['SSA', 'DHOA', 'EHO', 'PFA', 'PROPOSED']
    Classifier = ['SVR', 'MPNN', 'EL', 'BiLSTM', 'RNN', 'ENSEMBLE', 'PROPOSED']
    for u in range(len(Eval_all)):


        Acc_Table = np.zeros((len(Algorithm)+len(Classifier), len(Terms)))
        for j in range(len(Algorithm) + len(Classifier)):
            for k in range(len(Terms)):
                Acc_Table[j, k] = Eval_all[u, 3, j, k]
        Table = PrettyTable()
        Table.add_column('TERMS', Terms[0:])
        for k in range(len(Algorithm)):
            Table.add_column(Algorithm[k], Acc_Table[k, :])
        print('-------------------------------------------------- Dataset-', u + 1, '-',
              'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)
        print()

        Table = PrettyTable()
        Table.add_column('TERMS', Terms[0:])
        for k in range(len(Classifier)):
            tab = Acc_Table[k+5, :]
            Table.add_column(Classifier[k], tab)
        print('-------------------------------------------------- Dataset-', u + 1, '-',
              'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)
        print()

# Journ_Plot_Results()
# Plot_Results()
# plot_results_Statistical()


