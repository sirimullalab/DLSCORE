FROM ubuntu:bionic

MAINTAINER Md Mahmudulla Hassan <mhassan@miners.utep.edu>

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
&& rm -rf /var/lib/apt/lists/*

WORKDIR app/
COPY autodock_vina_1_1_2_linux_x86/ ./autodock_vina_1_1_2_linux_x86/
COPY mgltools/ ./mgltools/ 
#COPY MGLTools-1.5.6/mgltools_x86_64Linux2_1.5.6/bin ./mgltools/
COPY networks/ ./networks/
COPY NNScore/ ./NNScore/
COPY samples/ ./samples/
COPY dlscore.py test_run.sh ./
COPY requirements.txt ./

RUN pip3 install --no-cache -r requirements.txt

# Install MGLTools
RUN curl -O http://mgltools.scripps.edu/downloads/downloads/tars/releases/REL1.5.6/mgltools_x86_64Linux2_1.5.6.tar.gz
RUN tar -xvf mgltools_x86_64Linux2_1.5.6.tar.gz && rm mgltools_x86_64Linux2_1.5.6.tar.gz

WORKDIR /app/mgltools_x86_64Linux2_1.5.6
RUN ./install.sh && rm -rf *.gz
RUN cat ./initMGLtools.sh >> ~/.bashrc
#RUN /bin/bash -c "source ./initMGLtools.sh"
WORKDIR /app

#ENV PATH="/app/mgltools:${PATH}"

#ENTRYPOINT ["python3", "dlscore.py"]

