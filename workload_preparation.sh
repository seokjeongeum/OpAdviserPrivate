mysql -ppassword -e"create database twitter;"
#mysql -ppassword -e"drop database sbrw;"
#mysql -ppassword -e"drop database sbread;"
#mysql -ppassword -e"drop database sbwrite;"
mysql -ppassword -e"create database sbrw;"
mysql -ppassword -e"create database sbread;"
mysql -ppassword -e"create database sbwrite;"
mysql -ppassword -e"set global max_connections=500;"

sysbench  \
    --db-driver=mysql  \
    --mysql-host=db  \
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
    --mysql-host=db  \
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
    --mysql-host=db  \
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

cp -r -v oltpbench_files/. /oltpbench
cd /oltpbench || exit
ant bootstrap
ant resolve
ant build
./oltpbenchmark -b twitter -c config/sample_twitter_config.xml  --create=true --load=true
