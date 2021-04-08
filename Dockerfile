FROM python:3.5-slim

MAINTAINER Md Mahmudulla Hassan <mhassan@miners.utep.edu>

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    curl \
&& rm -rf /var/lib/apt/lists/*

WORKDIR app/
COPY autodock_vina_1_1_2_linux_x86/ ./autodock_vina_1_1_2_linux_x86/
COPY mgltools/ ./mgltools/ 
COPY NNScore/ ./NNScore/
COPY samples/ ./samples/
COPY dlscore.py test_run.sh ./
COPY requirements.txt ./

RUN pip install --no-cache -r requirements.txt

# Get the network files
RUN mkdir networks
RUN curl -o general.tar.gz https://drugdiscovery.utep.edu/download.php?file=general-10.tar.gz
RUN curl -o refined.tar.gz https://drugdiscovery.utep.edu/download.php?file=refined-10.tar.gz
RUN tar -xvf general.tar.gz && mv general networks/ && rm -f general.tar.gz
RUN tar -xvf refined.tar.gz && mv refined networks/ && rm -f refined.tar.gz

# Install MGLTools
RUN curl -O http://mgltools.scripps.edu/downloads/downloads/tars/releases/REL1.5.6/mgltools_x86_64Linux2_1.5.6.tar.gz
RUN tar -xvf mgltools_x86_64Linux2_1.5.6.tar.gz && rm mgltools_x86_64Linux2_1.5.6.tar.gz

WORKDIR /app/mgltools_x86_64Linux2_1.5.6
RUN ./install.sh && rm -rf *.gz
RUN cat ./initMGLtools.sh >> ~/.bashrc
RUN /bin/bash -c "source ./initMGLtools.sh"
WORKDIR /app

ENV PATH="/app/mgltools_x86_64Linux2_1.5.6/bin:${PATH}"

ENTRYPOINT ["python3", "dlscore.py"]

