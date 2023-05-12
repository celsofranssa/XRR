# activate venv and set Python path
source ~/projects/venvs/XRR/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/XRR/

data=Eurlex-4k
fold_idx=0

time_start=$(date '+%Y-%m-%d %H:%M:%S')
python main.py \
  tasks=[preprocess] \
  trainer.precision=16 \
  model=ULSE \
  data=$data \
  data.batch_size=64 \
  data.num_workers=12 \
  data.folds=[$fold_idx]
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > preprocess_time.txt

time_start=$(date '+%Y-%m-%d %H:%M:%S')
python main.py \
  tasks=[apmi] \
  trainer.precision=16 \
  model=ULSE \
  data=$data \
  data.batch_size=64 \
  data.num_workers=12 \
  data.folds=[$fold_idx]

python main.py \
  tasks=[fit,predict] \
  trainer.precision=16 \
  trainer.max_epochs=3 \
  trainer.max_epochs=1 \
  model=ULSE \
  data=$data \
  data.batch_size=64 \
  data.num_workers=12 \
  data.folds=[$fold_idx]
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > retrieve_time.txt

time_start=$(date '+%Y-%m-%d %H:%M:%S')
python main.py \
  tasks=[fit] \
  trainer.precision=16 \
  trainer.max_epochs=4 \
  trainer.max_epochs=2 \
  model=XRR \
  data=$data \
  data.batch_size=64 \
  data.num_workers=12 \
  data.folds=[$fold_idx]
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > fit_time.txt

time_start=$(date '+%Y-%m-%d %H:%M:%S')
python main.py \
  tasks=[predict] \
  trainer.precision=16 \
  model=XRR \
  data=$data \
  data.batch_size=64 \
  data.num_workers=12 \
  data.folds=[$fold_idx]
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > predict_time.txt

time_start=$(date '+%Y-%m-%d %H:%M:%S')
python main.py \
  tasks=[eval] \
  trainer.precision=16 \
  model=XRR \
  data=$data \
  data.batch_size=64 \
  data.num_workers=12 \
  data.folds=[$fold_idx]
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > eval_time.txt