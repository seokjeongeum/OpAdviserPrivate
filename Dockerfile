FROM jeongeumseok/opadviser
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt update && apt-get --no-install-recommends install -y  \
    mysql-client-5.7 \
    sysbench \
    git  \
    default-jdk \
    ant \
    python3.8  \
    python3.8-dev  \
    python3.8-venv  \
    python3-pip  \
    python3-setuptools  \
    build-essential

RUN rm -rf oltpbench
RUN git clone https://github.com/oltpbenchmark/oltpbench.git
COPY /oltpbench_files /oltpbench
WORKDIR /oltpbench
RUN ant bootstrap
RUN ant resolve
RUN ant build

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN python -m pip install pip
RUN python -m pip install --upgrade pip
WORKDIR /app
# By copying over requirements first, we make sure that Docker will cache
# our installed requirements rather than reinstall them on every build
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r requirements.txt
# Now copy in our code, and run it
COPY . /app
