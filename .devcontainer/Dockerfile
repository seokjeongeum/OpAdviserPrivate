FROM ubuntu:18.04
# Create custom resolv.conf during build
RUN echo "nameserver 141.223.1.2" > /etc/resolv.conf.override && \
    echo "nameserver 1.0.0.1" >> /etc/resolv.conf.override && \
    cat /etc/resolv.conf.override > /etc/resolv.conf
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN apt update && apt-get --no-install-recommends install -y  \
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
    sudo

RUN rm -rf sysbench
RUN git clone https://github.com/akopytov/sysbench.git && \
    cd sysbench && \
    git checkout ead2689ac6f61c5e7ba7c6e19198b86bd3a51d3c && \
    ./autogen.sh && \
    ./configure && \
    make && make install

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN python -m pip install pip
