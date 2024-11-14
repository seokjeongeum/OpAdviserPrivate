cd ~/jeseok || return
rm -r join-order-benchmark
git clone https://github.com/seokjeongeum/join-order-benchmark.git
cd join-order-benchmark || return
mv ./csv_files/schematext.sq ./csv_files/schematext_backup.sq
wget http://homepages.cwi.nl/~boncz/job/imdb.tgz
sudo tar -zxvf imdb.tgz -C csv_files
mv ./csv_files/schematext_backup.sq ./csv_files/schematext.sq
source csv_files/schematext.sql
source csv_files/fkindexes.sql
./load_data_mysql.sh
