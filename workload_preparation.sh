apt update
apt install mysql-server-5.7
echo "[mysqld]
innodb_log_checksums = 0">>/etc/my.cnf
service mysql start
mysql -uroot -e"ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'password';"
mysql -uroot -ppassword -e"set global max_connections=500;"


mysql -uroot -ppassword -e"drop database tatp;"
mysql -uroot -ppassword -e"create database tatp;"
/oltpbench/oltpbenchmark -b tatp -c /oltpbench/config/sample_tatp_config.xml  --create=true --load=true
mysql -uroot -ppassword -e"drop database tpcc;"
mysql -uroot -ppassword -e"create database tpcc;"
/oltpbench/oltpbenchmark -b tpcc -c /oltpbench/config/sample_tpcc_config.xml  --create=true --load=true
mysql -uroot -ppassword -e"drop database twitter;"
mysql -uroot -ppassword -e"create database twitter;"
/oltpbench/oltpbenchmark -b twitter -c /oltpbench/config/sample_twitter_config.xml  --create=true --load=true
mysql -uroot -ppassword -e"drop database voter;"
mysql -uroot -ppassword -e"create database voter;"
/oltpbench/oltpbenchmark -b voter -c /oltpbench/config/sample_voter_config.xml  --create=true --load=true
mysql -uroot -ppassword -e"drop database wikipedia;"
mysql -uroot -ppassword -e"create database wikipedia;"
/oltpbench/oltpbenchmark -b wikipedia -c /oltpbench/config/sample_wikipedia_config.xml  --create=true --load=true
mysql -uroot -ppassword -e"drop database ycsb;"
mysql -uroot -ppassword -e"create database ycsb;"
/oltpbench/oltpbenchmark -b ycsb -c /oltpbench/config/sample_ycsb_config.xml  --create=true --load=true

mysql -uroot -ppassword -e"drop database sbrw;"
mysql -uroot -ppassword -e"create database sbrw;"
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

mysql -uroot -ppassword -e"drop database sbread;"
mysql -uroot -ppassword -e"create database sbread;"
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

mysql -uroot -ppassword -e"drop database sbwrite;"
mysql -uroot -ppassword -e"create database sbwrite;"
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
