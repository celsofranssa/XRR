# activate venv and set Python path
source ~/projects/venvs/XRR/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/XRR/

python main.py \
  tasks=[fit] \
  trainer.precision=16 \
  model=XRR \
  data=Wiki10-31k \
  data.folds=[0]