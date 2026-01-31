import numpy as np
from Evaluation import error_evaluation
from Global_Vars import Global_Vars
from Model_Ensemble import Model_Ensemble


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
            Eval, predict = Model_Ensemble(Feat, Tar, sol)
            Eval = error_evaluation(predict, Test_Target)
            Fitn[i] = 1/(Eval[4] + Eval[7])
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Feat[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, predict = Model_Ensemble(Feat, Tar, sol)
        Eval = error_evaluation(predict, Test_Target)
    Fitn = 1 / (Eval[4] + Eval[7])
    return Fitn




















