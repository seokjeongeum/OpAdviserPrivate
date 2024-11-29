apt update
apt install -y \
  mysql-server-5.7 \
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
  sudo \

echo '[mysqld]
port=3308
innodb_log_checksums = 0' | sudo tee -a /etc/mysql/my.cnf
service mysql start 
mysql -e"ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'password';"
mysql -ppassword -e"CREATE USER 'root'@'127.0.0.1' IDENTIFIED BY 'password';"
mysql -ppassword -e"CREATE USER 'root'@'::1' IDENTIFIED BY 'password';"
mysql -ppassword -e"GRANT ALL PRIVILEGES ON *.* TO 'root'@'127.0.0.1';"
mysql -ppassword -e"GRANT ALL PRIVILEGES ON *.* TO 'root'@'::1';"
mysql -ppassword -e"FLUSH PRIVILEGES;"
mysql -ppassword -e"set global max_connections=100000;"
update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
python -m pip install --upgrade pip
pip install --user --upgrade setuptools
pip install --upgrade wheel
python -m pip install -r requirements.txt

cd /
rm -rf sysbench
git clone https://github.com/akopytov/sysbench.git && \
    cd sysbench && \
    git checkout ead2689ac6f61c5e7ba7c6e19198b86bd3a51d3c && \
    ./autogen.sh && \
    ./configure && \
    make && make install
mysql -ppassword -e"drop database sbrw;"
mysql -ppassword -e"create database sbrw;"
sysbench  \
    --db-driver=mysql  \
    --mysql-host=localhost  \
    --mysql-port=3308  \
    --mysql-user=root  \
    --mysql-password=password  \
    --table_size=800000  \
    --tables=300  \
    --events=0  \
    --threads=80  \
    --mysql-db=sbrw  \
    oltp_read_write  \
    prepare
cd ~/OpAdviserPrivate
export PYTHONPATH="."
python scripts/optimize.py --dbname=sbrw --workload=sysbench --workload_type=readwrite
