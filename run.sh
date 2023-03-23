# activate venv and set Python path
source ~/projects/venvs/xCoRetriv/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoRetriv/

python main.py \
  tasks=[fit] \
  trainer.precision=16 \
  model=RerankerBERT \
  ranking.retriever=BM25 \
  data=Amazon-670k \
  data.folds=[0]