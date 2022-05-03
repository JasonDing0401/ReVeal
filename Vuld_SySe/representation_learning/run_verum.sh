#python api_test.py --dataset chrome_debian --features wo_ggnn;
#for l in 0 5 2 4 3; do
#  python api_test.py --dataset chrome_debian/imbalanced --features ggnn --num_layers $l;
#done

#python api_test.py --dataset chrome_debian/imbalanced --features ggnn;
#python api_test.py --dataset chrome_debian/balanced --features ggnn;
#python api_test.py --dataset chrome_debian --features wo_ggnn;
#python api_test.py --dataset devign --features wo_ggnn;
#python api_test.py --dataset chrome_debian/imbalanced --features ggnn;
#python api_test.py --dataset devign --features ggnn;
#python api_test.py --dataset chrome_debian/balanced --features ggnn --baseline --baseline_balance --baseline_model svm;
#python api_test.py --dataset chrome_debian/balanced --features ggnn --baseline --baseline_balance --baseline_model lr;
#python api_test.py --dataset chrome_debian/balanced --features ggnn --baseline --baseline_balance --baseline_model rf;
#python api_test.py --dataset devign --features ggnn --baseline --baseline_balance --baseline_model svm;
#python api_test.py --dataset devign --features ggnn --baseline --baseline_balance --baseline_model lr;
#python api_test.py --dataset devign --features ggnn --baseline --baseline_balance --baseline_model rf;
for l in 10 15 20 25 30; do
python api_test.py --dataset chrome_debian/balanced --features ggnn --max_patience $l;
done
