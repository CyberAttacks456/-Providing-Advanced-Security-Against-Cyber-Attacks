import numpy as np
from Evaluation import error_evaluation
from Global_Vars import Global_Vars
from Model_MDS_ATCN import Model_MDS_ATCN


def objfun_feat(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :25]).astype(np.int16)
            w = np.round(Soln[i, 25]).astype(np.int16) # Weight
            feat = w * Feat[:, sol]
            varience = np.var(feat)
            Fitn[i] = 1 / varience
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        feat = Feat[:, sol]
        varience = np.var(feat)
        Fitn = 1 / varience
        return Fitn


def objfun_Pred(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Tar = np.reshape(Tar, (-1, 1))
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = Feat[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = Feat[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, predict = Model_MDS_ATCN(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval = error_evaluation(predict, Test_Target)
            Fitn[i] = 1 / (Eval[4] + Eval[7])
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Feat[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, predict = Model_MDS_ATCN(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval = error_evaluation(predict, Test_Target)
    Fitn = 1 / (Eval[4] + Eval[7])
    return Fitn
