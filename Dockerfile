FROM tensorflow/tensorflow:1.5.0-gpu
RUN apt-get update && apt-get install -y libsm6 libxrender-dev libsdl-dev git cmake && pip install opencv-python
WORKDIR /src
RUN git clone https://github.com/miyosuda/Arcade-Learning-Environment.git && cd Arcade-Learning-Environment && cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF . && make -j 4 && pip install .
CMD bash
