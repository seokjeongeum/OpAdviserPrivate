apt update && apt-get --no-install-recommends install -y mysql-server-5.7
service mysql start
mysql -e"ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'password';"
mysql -ppassword -e"create database tatp;"
mysql -ppassword -e"create database tpcc;"
mysql -ppassword -e"create database twitter;"
mysql -ppassword -e"create database voter;"
mysql -ppassword -e"create database wikipedia;"
mysql -ppassword -e"create database ycsb;"
mysql -ppassword -e"create database sbrw;"
mysql -ppassword -e"create database sbread;"
mysql -ppassword -e"create database sbwrite;"
mysql -ppassword -e"set global max_connections=500;"

cp -r -v oltpbench_files/. /oltpbench
cd /oltpbench || exit
ant bootstrap
ant resolve
ant build
./oltpbenchmark -b tatp -c config/sample_tatp_config.xml  --create=true --load=true
./oltpbenchmark -b tpcc -c config/sample_tpcc_config.xml  --create=true --load=true
./oltpbenchmark -b twitter -c config/sample_twitter_config.xml  --create=true --load=true
./oltpbenchmark -b voter -c config/sample_voter_config.xml  --create=true --load=true
./oltpbenchmark -b wikipedia -c config/sample_wikipedia_config.xml  --create=true --load=true
./oltpbenchmark -b ycsb -c config/sample_ycsb_config.xml  --create=true --load=true

sysbench  \
    --db-driver=mysql  \
    --mysql-host=localhost  \
    --mysql-port=3306  \
    --mysql-user=root  \
    --mysql-password=password  \
    --table_size=800000  \
    --tables=300  \
    --events=0  \
    --threads=32  \
    --mysql-db=sbrw  \
    oltp_read_write  \
    prepare

sysbench  \
    --db-driver=mysql  \
    --mysql-host=localhost  \
    --mysql-port=3306  \
    --mysql-user=root  \
    --mysql-password=password  \
    --table_size=800000  \
    --tables=300  \
    --events=0  \
    --threads=32  \
    --mysql-db=sbread  \
    oltp_read_only  \
    prepare

sysbench  \
    --db-driver=mysql  \
    --mysql-host=localhost  \
    --mysql-port=3306  \
    --mysql-user=root  \
    --mysql-password=password  \
    --table_size=800000  \
    --tables=300  \
    --events=0  \
    --threads=32  \
    --mysql-db=sbwrite  \
    oltp_write_only  \
    prepare
