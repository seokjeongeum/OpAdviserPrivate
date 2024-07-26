FROM ubuntu:18.04
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt update && apt-get --no-install-recommends install -y  \
    git  \
    default-jdk \
    ant \
    build-essential \
    openssh-client \
    cgroup-tools \
    libaio1 \
    libaio-dev \
    python3.8  \
    python3.8-dev  \
    python3.8-venv  \
    python3-pip  \
    python3-setuptools \
    autoconf \
    pkg-config \
    libtool \
    libmysqlclient-dev \
    automake \
    mysql-server-5.7

RUN rm -rf oltpbench && \
    git clone https://github.com/oltpbenchmark/oltpbench.git
COPY oltpbench_files /oltpbench
RUN cd /oltpbench && \
    ant bootstrap&& \
    ant resolve&& \
    ant build

RUN rm -rf sysbench
RUN git clone https://github.com/akopytov/sysbench.git

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN python -m pip install pip