#### seqfirstpos
#### train using docker with gpu
sudo nvidia-docker run -it --rm -v /your/workplace/seq/:/models/ tensorflow/tensorflow:1.9.0-gpu python /models/src/seq.py /models/data/traindata.ID
#### request by curl shell
curl -XPOST http://IP:port/v1/models/seq_model:predict -d "{\"signature_name\":\"predictByRaw\", \"instances\":[{\"input\":[\"aaaaa\",\"bbbbbb\",\"cccc\",\"ddddd\",\"None\",\"None\",\"None\",\"None\",\"None\",\"ddddd\",\"None\",\"None\",\"None\",\"dddddd\",\"None\",\"None\",\"None\",\"xxxx\",\"None\",\"None\",\"None\",\"xxxxxx\",\"None\",\"None\",\"None\",\"eeeeee\",\"None\",\"None\",\"None\",\"zzzzzz\",\"None\",\"None\",\"None\",\"eeeeee\",\"None\",\"None\",\"None\",\"hhhhhh\",\"None\",\"None\",\"None\"]}]}"
