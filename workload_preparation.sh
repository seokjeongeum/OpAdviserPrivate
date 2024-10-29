#run this in root user
apt update
apt install mysql-server-5.7 -y
echo "[mysqld]
innodb_log_checksums = 0">>/etc/my.cnf
service mysql start
mysql -e"ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'password';"
mysql -ppassword -e"set global max_connections=500;"
mysql -ppassword -e"CREATE USER 'root'@'127.0.0.1' IDENTIFIED BY 'password';
CREATE USER 'root'@'::1' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON *.* TO 'root'@'127.0.0.1';
GRANT ALL PRIVILEGES ON *.* TO 'root'@'::1';
FLUSH PRIVILEGES;"

mysql -ppassword -e"drop database sbrw;"
mysql -ppassword -e"create database sbrw;"
sysbench  \
    --db-driver=mysql  \
    --mysql-host=localhost  \
    --mysql-port=3306  \
    --mysql-user=root  \
    --mysql-password=password  \
    --table_size=800000  \
    --tables=300  \
    --events=0  \
    --threads=80  \
    --mysql-db=sbrw  \
    oltp_read_write  \
    prepare

mysql -ppassword -e"drop database sbread;"
mysql -ppassword -e"create database sbread;"
sysbench  \
    --db-driver=mysql  \
    --mysql-host=localhost  \
    --mysql-port=3306  \
    --mysql-user=root  \
    --mysql-password=password  \
    --table_size=800000  \
    --tables=300  \
    --events=0  \
    --threads=80  \
    --mysql-db=sbread  \
    oltp_read_only  \
    prepare

mysql -ppassword -e"drop database sbwrite;"
mysql -ppassword -e"create database sbwrite;"
sysbench  \
    --db-driver=mysql  \
    --mysql-host=localhost  \
    --mysql-port=3306  \
    --mysql-user=root  \
    --mysql-password=password  \
    --table_size=800000  \
    --tables=300  \
    --events=0  \
    --threads=80  \
    --mysql-db=sbwrite  \
    oltp_write_only  \
    prepare

cp -r -v oltpbench_files/. /oltpbench/
cd /oltpbench || exit
ant bootstrap
ant resolve
ant build
mysql -ppassword -e"drop database resourcestresser;"
mysql -ppassword -e"create database resourcestresser;"
/oltpbench/oltpbenchmark -b resourcestresser -c /oltpbench/config/sample_resourcestresser_config.xml  --create=true --load=true
mysql -ppassword -e"drop database seats;"
mysql -ppassword -e"create database seats;"
/oltpbench/oltpbenchmark -b seats -c /oltpbench/config/sample_seats_config.xml  --create=true --load=true
mysql -ppassword -e"drop database sibench;"
mysql -ppassword -e"create database sibench;"
/oltpbench/oltpbenchmark -b sibench -c /oltpbench/config/sample_sibench_config.xml  --create=true --load=true
mysql -ppassword -e"drop database smallbank;"
mysql -ppassword -e"create database smallbank;"
/oltpbench/oltpbenchmark -b smallbank -c /oltpbench/config/sample_smallbank_config.xml  --create=true --load=true
mysql -ppassword -e"drop database tatp;"
mysql -ppassword -e"create database tatp;"
/oltpbench/oltpbenchmark -b tatp -c /oltpbench/config/sample_tatp_config.xml  --create=true --load=true
mysql -ppassword -e"drop database tpcc;"
mysql -ppassword -e"create database tpcc;"
/oltpbench/oltpbenchmark -b tpcc -c /oltpbench/config/sample_tpcc_config.xml  --create=true --load=true
mysql -ppassword -e"drop database twitter;"
mysql -ppassword -e"create database twitter;"
/oltpbench/oltpbenchmark -b twitter -c /oltpbench/config/sample_twitter_config.xml  --create=true --load=true
mysql -ppassword -e"drop database voter;"
mysql -ppassword -e"create database voter;"
/oltpbench/oltpbenchmark -b voter -c /oltpbench/config/sample_voter_config.xml  --create=true --load=true
mysql -ppassword -e"drop database wikipedia;"
mysql -ppassword -e"create database wikipedia;"
/oltpbench/oltpbenchmark -b wikipedia -c /oltpbench/config/sample_wikipedia_config.xml  --create=true --load=true
mysql -ppassword -e"drop database ycsb;"
mysql -ppassword -e"create database ycsb;"
/oltpbench/oltpbenchmark -b ycsb -c /oltpbench/config/sample_ycsb_config.xml  --create=true --load=true
