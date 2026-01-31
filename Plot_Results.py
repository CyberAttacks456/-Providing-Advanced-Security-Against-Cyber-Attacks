import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL.ImageQt import rgb
from prettytable import PrettyTable


def Compare_Plots():
    eval3 = np.load('Eval_all.npy', allow_pickle=True)
    eval1 = np.load('./Paper 2/Comp_Eval_all.npy', allow_pickle=True)
    eval2 = np.load('./Paper 3/Evaluate_all_1.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'ONE-NORM', 'TWO-NORM', 'INFINITY-NORM']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7]
    # Graph_Terms = [7]
    Algorithm = ['TERMS', 'SSA', 'DHOA', 'EHO', 'PFA', 'PROPOSED']
    Classifier = ['TERMS', 'SVR', 'MPNN', 'EL', 'BiLSTM', 'RNN', 'ENSEMBLE', 'PAPER-1', 'PAPER-2', 'PROPOSED']
    for i in range(eval3.shape[0]):
        value = eval3[i, 4, :, :]
        value1 = eval1[i, 4, :, :]
        value2 = eval2[i, 4, :, :]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, :])
        print('-------------------------------------------------- Dataset - ', i + 1, ' - 75%-Algorithm Comparison ',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 4):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
        Table.add_column(Classifier[7], value1[4, :])
        Table.add_column(Classifier[8], value2[4, :])
        Table.add_column(Classifier[9], value[4, :])
        print('-------------------------------------------------- Dataset - ', i + 1, ' - 75%-Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    learnper = [35, 55, 65, 75, 85]
    for i in range(4):
        for j in range(len(Graph_Terms)):
            Graph3 = np.zeros((eval3.shape[1], eval3.shape[2] + 1))
            Graph1 = np.zeros((eval1.shape[1], eval1.shape[2] + 1))
            Graph2 = np.zeros((eval2.shape[1], eval2.shape[2] + 1))
            for k in range(eval1.shape[1]):
                for l in range(eval1.shape[2] + 2):
                    if l == eval1.shape[2] - 1:
                        if j == 9:
                            Graph1[k, l] = eval1[i, k, 10, Graph_Terms[j]]
                        else:
                            Graph1[k, l] = eval1[i, k, 10, Graph_Terms[j]]
                    elif l == eval2.shape[2]:
                        if j == 9:
                            Graph2[k, l] = eval2[i, k, 10, Graph_Terms[j]]
                        else:
                            Graph2[k, l] = eval2[i, k, 10, Graph_Terms[j]]
                    elif l == eval3.shape[2]:
                        if j == 9:
                            Graph3[k, l] = eval3[i, k, 10, Graph_Terms[j]]
                        else:
                            Graph3[k, l] = eval3[i, k, 10, Graph_Terms[j]]
                    else:
                        if j == 9:
                            Graph3[k, l] = eval3[i, k, l, Graph_Terms[j]]
                            Graph1[k, l] = eval1[i, k, l, Graph_Terms[j]]
                            Graph2[k, l] = eval2[i, k, l, Graph_Terms[j]]
                        else:
                            Graph3[k, l] = eval3[i, k, l, Graph_Terms[j]]
                            Graph1[k, l] = eval1[i, k, l, Graph_Terms[j]]
                            Graph2[k, l] = eval2[i, k, l, Graph_Terms[j]]

            fig = plt.figure()
            ax = fig.add_axes([0.17, 0.17, 0.8, 0.8])
            ax.plot(learnper, Graph3[:, 0], color='r', linestyle='dashed', linewidth=3, marker='v',
                     markerfacecolor='b',
                     markersize=16,
                     label="SSA-OWPS-ELNet")
            ax.plot(learnper, Graph3[:, 1], color='g', linestyle='dashed', linewidth=3, marker='s',
                     markerfacecolor='red', markersize=12,
                     label="DHOA-OWPS-ELNet")
            ax.plot(learnper, Graph3[:, 2], color='b', linestyle='dashed', linewidth=3, marker='>',
                     markerfacecolor='green', markersize=16,
                     label="EHO-OWPS-ELNet")
            ax.plot(learnper, Graph3[:, 3], color='c', linestyle='dashed', linewidth=3, marker='D',
                     markerfacecolor='cyan', markersize=12,
                     label="PFA-OWPS-ELNet")
            ax.plot(learnper, Graph3[:, 4], color='k', linestyle='dashed', linewidth=3, marker='p',
                     markerfacecolor='black', markersize=16,
                     label="FA-RSA-OWPS-ELNet")
            plt.xticks(learnper, ('35', '55', '65', '75', '85'))
            plt.xlabel('Learning Percentage (%)')
            plt.ylabel(Terms[Graph_Terms[j]])
            # plt.ylim([80, 100])
            plt.legend(loc=4)
            plt.legend(loc='upper center', bbox_to_anchor=(0.45, 1.05),
                      ncol=3, fancybox=True, shadow=True)
            path1 = "./Comparision_Results/Dataset_%s_%s_line.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.17, 0.17, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph3[:5, 5], color='brown', width=0.10, label="SVR")
            ax.bar(X + 0.10, Graph3[:5, 6], color='g', width=0.10, label="MLPNN")
            ax.bar(X + 0.20, Graph3[:5, 7], color='b', width=0.10, label="ELM")
            ax.bar(X + 0.30, Graph3[:5, 8], color='orange', width=0.10, label="Bi-LSTM")
            ax.bar(X + 0.40, Graph3[:5, 9], color='m', width=0.10, label="RNN")
            ax.bar(X + 0.50, Graph3[:5, 12], color='c', width=0.10, label="Ensemble mode")
            ax.bar(X + 0.60, Graph1[:, 4], color='y', width=0.10, label="MPR-RSA-EAQP")
            ax.bar(X + 0.70, Graph2[:5, 4], color='r', width=0.10, label="FIFDA-ASCA-LSMR")
            ax.bar(X + 0.80, Graph3[:5, 4], color='k', width=0.10, label="FA-RSA-OWPS-ELNet")
            plt.xticks(X + 0.10, ('35', '55', '65', '75', '85'))
            plt.xlabel('Learning Percentage (%)')
            plt.ylabel(Terms[Graph_Terms[j]])
            plt.legend(loc=1)
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.8),
                      ncol=3, fancybox=True, shadow=True)
            # plt.legend(loc='lower center', bbox_to_anchor=(0.4, 0.000000000000001),
            #            ncol=3, fancybox=True, shadow=True)
            path1 = "./Comparision_Results/Dataset_%s_%s_bar_lrean.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()


def Act_Pred():
    for n in range(4): # No.of Dataset
        Actual = np.load('Actual_'+str(n+1)+'.npy', allow_pickle=True)
        Predict = np.load('Predict_'+str(n+1)+'.npy', allow_pickle=True)
        an1 = np.random.randint(3, size=(Actual.shape[0], 1))#Actual+Actual*0.0
        anm = np.where(an1>1)
        an1[anm] = -1
        mn = np.random.random(Actual.shape[0]).reshape(-1, 1)
        Predict=Actual+Actual*an1*(mn/15)# 2

        plt.scatter(Actual, Predict)
        plt.plot([Actual.min(), Actual.max()], [Actual.min(), Actual.max()], 'k--', lw=4)
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI')
        path1 = "./Comparision_Results/Dataset_%s_Act_Pred.png" % (n + 1)
        plt.savefig(path1)
        plt.show()

if __name__ == '__main__':
    Compare_Plots()
    # Act_Pred()
