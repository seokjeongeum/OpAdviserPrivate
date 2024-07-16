mysql -hdb -ppassword -e"create database sbrw;"
mysql -hdb -ppassword -e"create database sbread;"
mysql -hdb -ppassword -e"create database sbwrite;"
sysbench  \
    --db-driver=mysql  \
    --mysql-host=db  \
    --mysql-socket=$SOCK  \
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
    --mysql-socket=$SOCK  \
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
    --mysql-socket=$SOCK  \
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

mysql -hdb -ppassword -e"create database twitter;"
mysql -hdb -ppassword -e"set global max_connections=500;"
/oltpbench/oltpbenchmark -b twitter -c /oltpbench/config/sample_twitter_config.xml  --create=true --load=true