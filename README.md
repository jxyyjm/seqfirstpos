### 1) train using docker with gpu
#### 1.1) pull a tf-gpu docker image
sudo nvidia-docker run -it --rm tensorflow/tensorflow:1.9.0-gpu
#### 1.2) train seq model using gpu
sudo nvidia-docker run -it --rm -v /your/workplace/seq/:/models/ tensorflow/tensorflow:1.9.0-gpu python /models/src/seq.py /models/data/traindata.ID

### 2) check tf-serving 
saved_model_cli show --dir=../seq_model/201912182207 --all

### 3) Serving on local machine
#### 3.1) pull a tf-serving images
sudo docker run -it --rm  docker.io/tensorflow/serving:latest
#### 3.2) start this serving on local
sudo docker run -e MODEL_NAME=seq_model -p 8500:8500 -it -v /your/model/dir/:/models/model/ TF-Serving-Image

### 3) serving on other machine
#### 3.1) run a tf-serving image
sudo docker run -d your-tf-serving-docker-image
#### 3.2) copy the model into this image
sudo docker cp ../seq_model/ ContainID:/models/
#### 3.3) save the image
sudo docker commit --change "ENV MODEL_NAME seq_model" ContainID NewImageName
#### 3.4) start this serving or use it some-where
sudo docker run -p 8500:8500 -it NewImageName

### 4) request 
#### 4.1) request by curl
curl -XPOST http://IP:port/v1/models/seq_model:predict -d "{\"signature_name\":\"predictByRaw\", \"instances\":[{\"input\":[\"aaaaa\",\"bbbbbb\",\"cccc\",\"ddddd\",\"None\",\"None\",\"None\",\"None\",\"None\",\"ddddd\",\"None\",\"None\",\"None\",\"dddddd\",\"None\",\"None\",\"None\",\"xxxx\",\"None\",\"None\",\"None\",\"xxxxxx\",\"None\",\"None\",\"None\",\"eeeeee\",\"None\",\"None\",\"None\",\"zzzzzz\",\"None\",\"None\",\"None\",\"eeeeee\",\"None\",\"None\",\"None\",\"hhhhhh\",\"None\",\"None\",\"None\"]}]}"
#### 4.2) request by http 
python RequestByHttp.py --args
#### 4.3) request by grpc
python RequestByGrpc.py --args

