FROM python:3.8-slim AS compile-image

ENV DEBIAN_FRONTEND=noninteractive
RUN BUILDPKGS="build-essential \
        libcurl4-openssl-dev \
        zlib1g-dev \
        libfftw3-dev \
        libc++-dev \
        git \
        wget \
        hdf5-tools \
        " && \
    apt-get update && \
    apt-get install -y --no-install-recommends apt-utils debconf locales locales-all && dpkg-reconfigure locales && \
    apt-get install -y --no-install-recommends $BUILDPKGS

RUN python -m venv /opt/venv
# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

# install dependencies:
COPY pycisTopic/requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir Cython numpy==1.20.3 && \
    pip install --no-cache-dir fitsne && \
    pip install --no-cache-dir papermill && \
    pip install --no-cache-dir igv_jupyterlab && \
    pip install --no-cache-dir bs4 && \
    pip install --no-cache-dir MACS2 && \
    pip install --no-cache-dir lxml && \
    pip install --no-cache-dir tspex && \
    pip install --no-cache-dir plotly && \
    pip install --no-cache-dir kaleido && \
    pip install --no-cache-dir pyvis && \
    pip install velocyto && \
    pip install --no-cache-dir pygam && \
    pip install --no-cache-dir fa2 && \
    pip install --no-cache-dir scanpy==1.8.2 && \
    pip install --no-cache-dir -r /tmp/requirements.txt 
    
# install pyscenic from local copy:
COPY pySCENIC /tmp/pySCENIC
RUN  cd /tmp/pySCENIC && \
     pip install . && \
     cd .. && rm -rf pySCENIC
    
# install pycisTopic from local copy:
COPY pycisTopic /tmp/pycisTopic
RUN  cd /tmp/pycisTopic && \
     pip install . && \
     cd .. && rm -rf pycisTopic

# install Mallet (https://github.com/mimno/Mallet)
# https://github.com/docker-library/openjdk/blob/0584b2804ed12dca7c5e264b5fc55fc07a3ac148/8-jre/slim/Dockerfile#L51-L54
RUN apt update
RUN mkdir -p /usr/share/man/man1 && \
    apt-get install -y --no-install-recommends ant openjdk-11-jdk && \
    git clone --depth=1 https://github.com/mimno/Mallet.git /tmp/Mallet && \
    cd /tmp/Mallet && \
    ant

# install pycistarget
COPY pycistarget /tmp/pycistarget
RUN cd /tmp/pycistarget && \
    pip install . && \
    cd .. && rm -rf pycistarget
    
# install scenicplus
COPY scenicplus /tmp/scenicplus
RUN cd /tmp/scenicplus && \
    pip install . && \
    cd .. && rm -rf scenicplus

RUN pip install polars

RUN pip install black

FROM python:3.8-slim AS build-image

RUN mkdir -p /usr/share/man/man1 && \
    apt-get -y update && \
    apt-get -y --no-install-recommends install \
        openjdk-11-jdk \
        procps \
        bash-completion \
        curl \
        libfftw3-dev \
        less && \
    rm -rf /var/cache/apt/* && \
    rm -rf /var/lib/apt/lists/*

COPY --from=compile-image /opt/venv /opt/venv
COPY --from=compile-image /tmp/Mallet /opt/mallet

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"
ENV PATH="/opt/mallet/bin:$PATH"
