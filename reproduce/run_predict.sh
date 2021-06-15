set -e
date
# extract features
# this may cost 4 ~ 5 hours
python extract_feature.py

# prediction
# this may cost 4 hours
cd pcqm4m_v1
python predict.py ../test_graph.pk ./models/model_0.068.ckpt
mv y_pred_pcqm4m.npz ../y_pred_pcqm4m1.npz
python predict.py ../test_graph.pk ./models/model_0.076.ckpt
mv y_pred_pcqm4m.npz ../y_pred_pcqm4m2.npz
rm -rf __pycache__ lightning_logs

date
cd ../pcqm4m_v2
python predict.py ../test_graph.pk ./models/model_0.055.ckpt
mv y_pred_pcqm4m.npz ../y_pred_pcqm4m3.npz
python predict.py ../test_graph.pk ./models/model_0.064.ckpt
mv y_pred_pcqm4m.npz ../y_pred_pcqm4m4.npz
rm -rf __pycache__ lightning_logs

date
cd ../pcqm4m_v3
python predict.py ../test_graph.pk ./models/model_0.068.ckpt
mv y_pred_pcqm4m.npz ../y_pred_pcqm4m5.npz
python predict.py ../test_graph.pk ./models/model_0.065.ckpt
mv y_pred_pcqm4m.npz ../y_pred_pcqm4m6.npz
rm -rf __pycache__ lightning_logs

cd ..

date
python ensemble_score.py

rm -rf y_pred_pcqm4m1.npz y_pred_pcqm4m2.npz y_pred_pcqm4m3.npz y_pred_pcqm4m4.npz y_pred_pcqm4m5.npz y_pred_pcqm4m6.npz
date
