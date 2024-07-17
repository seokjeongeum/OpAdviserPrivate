FROM ubuntu:18.04
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt update && apt-get --no-install-recommends install -y  \
    mysql-server-5.7 \
    sysbench \
    git  \
    default-jdk \
    ant \
    python3.8  \
    python3.8-dev  \
    python3.8-venv  \
    python3-pip  \
    python3-setuptools  \
    build-essential \
    openssh-client \
    sudo \
    cgroup-tools

RUN addgroup --gid 1000 devcontainer && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos "" devcontainer && \
    echo 'devcontainer ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

RUN service mysql start && \
mysql -e"ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'password';" && \
mysql -ppassword -e"create database twitter;" && \
mysql -ppassword -e"create database sbrw;" && \
mysql -ppassword -e"create database sbread;" && \
mysql -ppassword -e"create database sbwrite;"

RUN rm -rf oltpbench
RUN git clone https://github.com/oltpbenchmark/oltpbench.git
COPY /oltpbench_files /oltpbench
WORKDIR /oltpbench
RUN ant bootstrap
RUN ant resolve
RUN ant build
COPY /oltpbench ~/oltpbench

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN python -m pip install pip
RUN python -m pip install --upgrade pip

ENTRYPOINT ["service mysql start"]
