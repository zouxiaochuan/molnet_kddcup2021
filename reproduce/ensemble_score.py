import numpy as np

import ogb
import ogb.lsc

if __name__ == '__main__':
    filenames = [
        'y_pred_pcqm4m1.npz',
        'y_pred_pcqm4m2.npz',
        'y_pred_pcqm4m3.npz',
        'y_pred_pcqm4m4.npz',
        'y_pred_pcqm4m5.npz',
        'y_pred_pcqm4m6.npz'
    ]

    scores = np.stack(
        [np.load(filename)['y_pred'] for filename in filenames])

    print(np.mean(np.abs(scores[0] - scores[1])))
    print(np.mean(np.abs(scores[0] - scores[2])))
    print(np.mean(np.abs(scores[0] - scores[3])))
    print(np.mean(np.abs(scores[3] - scores[2])))
    print(np.mean(np.abs(scores[4] - scores[3])))
    print(np.mean(np.abs(scores[4] - scores[0])))
    print(np.mean(np.abs(scores[4] - scores[5])))

    print('-----')
    
    score1 = np.mean(scores[[0, 1]], axis=0)
    score2 = np.mean(scores[[2, 3]], axis=0)
    score3 = np.mean(scores[[4, 5]], axis=0)

    print(np.mean(np.abs(score1 - score2)))
    print(np.mean(np.abs(score2 - score3)))
    print(np.mean(np.abs(score1 - score3)))

    mean_score = (score1 + score2 + score3) / 3
    print(score1)
    print(score2)
    print(score3)
    print(mean_score)
    
    evaluator = ogb.lsc.PCQM4MEvaluator()
    input_dict = {'y_pred': mean_score}
    
    evaluator.save_test_submission(
        input_dict = input_dict, dir_path = './')

    pass
