This repository is used for submitting code of team no_free_lunch for pcqm4m track of ogb-lsc

this repository has two parts:
1. /pcqm4m is a clean version of MolNet
2. /reproduce is used for reproducing test score

## how to reproduce test score
Due to the file size limitation of github, model files cannot be saved to github, you can download full package by https://kddcup2021.s3.ap-southeast-1.amazonaws.com/pcqm4m_fin.tar
and then
enter /reproduce directory and run "./run_predict.sh", it will take about 8 hours to run on single core single card machine.
If you have any problems of downloading the full package, please contact me.


## how to do training
1. first download pcqm4m dataset from the official site.
2. enter /pcqm4m and run "python create_dataset_new.py ../../datasets" to generate dataset. ../../datasets is where you put pcqm4m dataset folder.
3. enter /pcqm4m and run "python run_molnet.py", it will use default dataset path of ../../datasets. And model is saved in lightning_logs directory.