# Failure Estimator of ASREvolve

Extract the failure estimator of ASREvolve ([original code](https://github.com/soarsmu/ASREvolve/blob/master/estimator/huggingface.py))

## Environment

```
# run a docker container
# alternatively you can use virtual environment
docker run --rm -it --name=estimator_0 --gpus '"device=0"' --shm-size 32G -it --mount type=bind,src=/media/mhilmiasyrofi/failure-estimator/,dst=/media/mhilmiasyrofi/failure-estimator/ pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

cd /media/mhilmiasyrofi/failure-estimator/

pip install -r requirements.txt
```
please modify `/media/mhilmiasyrofi/failure-estimator/` accordingly

## Dummy data for training

For experimental use, you can use `dummy_corpus.json`.

To match with the format data from ASRDebugger, the training data for the failure estimator is a json file. For each line, it contains `label` and `text` value. `label:1` indicates for a failed test case and `label:0` indicates for a non failed test case.

## Try

```
cd /media/mhilmiasyrofi/failure-estimator/
python main.py
```

it will output an example of 10 randomly selected samples. Where it predict the fail probabilty of the label
```
label: 0, prob: 0.02
label: 0, prob: 0.02
label: 0, prob: 0.03
label: 0, prob: 0.02
label: 1, prob: 0.91
label: 0, prob: 0.03
label: 1, prob: 0.91
label: 1, prob: 0.94
label: 0, prob: 0.04
label: 0, prob: 0.03
```





