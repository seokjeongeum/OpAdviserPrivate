#!/bin/bash

cd ~/jeseok || exit
rm -r join-order-benchmark
git clone https://github.com/seokjeongeum/join-order-benchmark.git
cd join-order-benchmark || exit
mv ./csv_files/schematext.sql ./csv_files/schematext_backup.sql
wget http://homepages.cwi.nl/~boncz/job/imdb.tgz
sudo tar -zxvf imdb.tgz -C csv_files
mv ./csv_files/schematext_backup.sql ./csv_files/schematext.sql
source csv_files/schematext.sql
source csv_files/fkindexes.sql
./load_data_mysql.sh
