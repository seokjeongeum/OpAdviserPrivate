echo "innodb_read_io_threads=4" >> /etc/my.cnf
echo "innodb_write_io_threads=8" >> /etc/my.cnf #To stress the double write buffer
echo "innodb_buffer_pool_size=20G" >> /etc/my.cnf # 70-80% available Memory
echo "innodb_buffer_pool_load_at_startup=ON" >> /etc/my.cnf
echo "innodb_log_file_size = 32M" >> /etc/my.cnf #Small log files, more page flush
echo "innodb_log_files_in_group=2" >> /etc/my.cnf
echo "innodb_file_per_table=1" >> /etc/my.cnf
echo "innodb_log_buffer_size=8M" >> /etc/my.cnf
echo "innodb_flush_method=O_DIRECT" >> /etc/my.cnf
echo "innodb_flush_log_at_trx_commit=0" >> /etc/my.cnf
echo "skip-innodb_doublewrite" >> /etc/my.cnf #commented or not depending on test

mysql -hdb -ppassword -e"create database twitter;"
mysql -hdb -ppassword -e"drop database sbrw;"
mysql -hdb -ppassword -e"drop database sbread;"
mysql -hdb -ppassword -e"drop database sbwrite;"
mysql -hdb -ppassword -e"create database sbrw;"
mysql -hdb -ppassword -e"create database sbread;"
mysql -hdb -ppassword -e"create database sbwrite;"
mysql -hdb -ppassword -e"set global max_connections=500;"

cp -r -v oltpbench_files/. /oltpbench
cd /oltpbench||exit
ant bootstrap
ant resolve
ant build
/oltpbench/oltpbenchmark -b twitter -c /oltpbench/config/sample_twitter_config.xml  --create=true --load=true

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
