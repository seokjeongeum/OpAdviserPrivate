#!/bin/bash

mkdir jeseok
cd jeseok || return

sudo yum update
sudo rpm --import https://repo.mysql.com/RPM-GPG-KEY-mysql
sudo wget -O /etc/pki/rpm-gpg/RPM-GPG-KEY-mysql https://repo.mysql.com/RPM-GPG-KEY-mysql-2022
wget https://dev.mysql.com/get/mysql57-community-release-el7-11.noarch.rpm
sudo rpm -ivh mysql57-community-release-el7-11.noarch.rpm
sudo yum localinstall mysql57-community-release-el7-11.noarch.rpm
sudo yum install -y \
mysql-devel \
gcc \
openssl-devel \
libffi-devel \
bzip2-devel \
zlib-devel \
mysql-community-server \
make \
automake \
libtool \
pkgconfig \
libaio-devel \
postgresql-devel \


cd /opt || return
sudo curl -O https://www.python.org/ftp/python/3.8.20/Python-3.8.20.tgz
sudo tar -zxvf Python-3.8.20.tgz
cd Python-3.8.20 || return
sudo ./configure --enable-shared
sudo make
sudo make install

cd ~/jeseok || return
rm -rf sysbench
git clone https://github.com/akopytov/sysbench.git
cd sysbench || return
git checkout ead2689ac6f61c5e7ba7c6e19198b86bd3a51d3c
./autogen.sh
sudo ./configure
sudo make
sudo make install

sudo systemctl set-environment MYSQLD_OPTS="--skip-grant-tables"
sudo systemctl start mysqld
sudo mysql -e"ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'password';"
sudo mysql -ppassword -e"CREATE USER 'root'@'127.0.0.1' IDENTIFIED BY 'password';
CREATE USER 'root'@'::1' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON *.* TO 'root'@'127.0.0.1';
GRANT ALL PRIVILEGES ON *.* TO 'root'@'::1';
FLUSH PRIVILEGES;"

cd ~/jeseok || return
rm -rf oltpbench
git clone https://github.com/oltpbenchmark/oltpbench.git
cd ~/jeseok/OpAdviserPrivate || return
cp -r oltpbench_files_cluster/. ~/oltpbench
export ANT_HOME=/usr/share/ant/
cd ~/jeseok/oltpbench || return
ant bootstrap
ant resolve
ant build

# Backup existing sshd_config
bak_file="/etc/ssh/sshd_config.bak"
if [ -f /etc/ssh/sshd_config ]; then
    cp /etc/ssh/sshd_config $bak_file
fi

# Remove or comment out existing timeout settings
sed -i.bak -e 's/^ClientAliveInterval/# ClientAliveInterval/' \
           -e 's/^ClientAliveCountMax/# ClientAliveCountMax/' \
           -e 's/^TCPKeepAlive/# TCPKeepAlive/' \
           /etc/ssh/sshd_config

# Add new timeout settings (if desired)
echo 'ClientAliveInterval 300' >> /etc/ssh/sshd_config
echo 'ClientAliveCountMax 3' >> /etc/ssh/sshd_config
echo 'TCPKeepAlive yes' >> /etc/ssh/sshd_config

# Notify user
echo "SSH server configuration modified to remove timeout settings."

cd ~/jeseok/OpAdviserPrivate || return
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install .
