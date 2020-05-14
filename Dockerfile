FROM kuberlab/pytorch:1.4.0-gpu

COPY ./models /
COPY ./requirements.txt /
RUN pip install -r /requirements.txt && cd /models/FAC/kernelconv2d && python setup.py develop && \
  cp kernelconv2d_cuda.* /usr/local/lib/python3.6/dist-packages/ && \
  rm -rf /requirements.txt /models
