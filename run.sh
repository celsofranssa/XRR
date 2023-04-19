# activate venv and set Python path
source ~/projects/venvs/XRR/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/XRR/

python main.py \
  tasks=[fit] \
  trainer.precision=16 \
  model=XRR \
  data=Amazon-670k \
  data.batch_size=64 \
  data.num_workers=12