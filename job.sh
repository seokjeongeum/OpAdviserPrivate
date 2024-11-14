git clone https://github.com/seokjeongeum/join-order-benchmark.git
cd join-order-benchmark || return
wget http://homepages.cwi.nl/~boncz/job/imdb.tgz
sudo tar -zxvf imdb.tgz -C csv_files
./load_data_mysql.sh
