# Run "docker cp devcontainer-opadviser-1:/var/lib/mysql /mnt/sdc/jeseok2/mysql2" in host
chown -R mysql:mysql /var/lib/mysql
service mysql start
mysql -e"ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'password';"
mysql -ppassword -e"set global max_connections=500;"

cp -r -v oltpbench_files/. /oltpbench
cd /oltpbench || exit
ant bootstrap
ant resolve
ant build
mysql -ppassword -e"drop database tatp;"
mysql -ppassword -e"create database tatp;"
./oltpbenchmark -b tatp -c config/sample_tatp_config.xml  --create=true --load=true
mysql -ppassword -e"create database tpcc;"
./oltpbenchmark -b tpcc -c config/sample_tpcc_config.xml  --create=true --load=true
mysql -ppassword -e"create database twitter;"
./oltpbenchmark -b twitter -c config/sample_twitter_config.xml  --create=true --load=true
mysql -ppassword -e"create database voter;"
./oltpbenchmark -b voter -c config/sample_voter_config.xml  --create=true --load=true
mysql -ppassword -e"create database wikipedia;"
./oltpbenchmark -b wikipedia -c config/sample_wikipedia_config.xml  --create=true --load=true
./oltpbenchmark -b ycsb -c config/sample_ycsb_config.xml  --create=true --load=true
mysql -ppassword -e"create database ycsb;"

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
    --threads=64  \
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
    --threads=64  \
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
    --threads=64  \
    --mysql-db=sbwrite  \
    oltp_write_only  \
    prepare
