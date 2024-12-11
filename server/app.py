import argparse
import time
from flask import Flask, request,jsonify
import mysql

from autotune.database.mysqldb import MysqlDB
from autotune.database.postgresqldb import PostgresqlDB
from autotune.dbenv import DBEnv
from autotune.tuner import DBTuner
from autotune.utils.config import parse_args
import os
from mysql.connector import pooling
import pprint
from flask_cors import CORS
dbconfig={
    'host':"localhost",
    'user':"root",
    'password':"password",
    'database':"concert_singer",
    'port':3307,
    }
pool=pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=32,
    pool_reset_session=True,
    **dbconfig
)
app = Flask(__name__)
CORS(app)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="scripts/demo.ini", help="config file"
)
parser.add_argument("run")
parser.add_argument("--no-debugger", action="store_false")
parser.add_argument("--no-reload", action="store_false")
opt = parser.parse_args()


args_db, args_tune = parse_args(opt.config)
if args_db["db"] == "mysql":
    db = MysqlDB(args_db)
elif args_db["db"] == "postgresql":
    db = PostgresqlDB(args_db)

env = DBEnv(args_db, args_tune, db)
tuner = DBTuner(args_db, args_tune, env)
print("Initialized")

queries=set()
@app.post("/query")
def query():
    query = request.get_json()  
    if query=='conduct tuning':
        if os.path.exists("/root/OpAdviserPrivate/autotune/cli/selectedList_demo.txt"):
            os.remove("/root/OpAdviserPrivate/autotune/cli/selectedList_demo.txt")
        for i, q in enumerate(queries):
            with open(
                f"/root/OpAdviserPrivate/autotune/demo_query/queries-mysql-new/{i}.sql", "w"
            ) as f:
                f.write(
                    f"""select current_timestamp(6) into @query_start;
set @query_name='{i}';
{q};
set @query_time_ms= timestampdiff(microsecond, @query_start, current_timestamp(6))/1000;
SELECT @query_name, @query_time_ms;
    """
                )
                with open('/root/OpAdviserPrivate/autotune/cli/selectedList_demo.txt','a')as f2:
                    f2.write(f'''{i}.sql
''')
        data=tuner.tune()
        return {
            "execution_times": data[1],
        }
    try:
        queries.add(query)
        mydb = pool.get_connection()
        cursor = mydb.cursor(dictionary=True)
        start_time = time.time()
        cursor.execute(query)
        end_time = time.time()
        data=cursor.fetchall()
        pprint.pprint(data)
        return {
            "data": data,
            "execution_time": end_time - start_time,
        }
    finally:
        if 'mydb' in locals():
            mydb.close()


@app.post("/knobs")
def knobs():
    try:
        mydb = pool.get_connection()
        cursor = mydb.cursor()
        cursor.execute("""
SHOW VARIABLES 
where variable_name!='optimizer_switch'
and variable_name!='sql_mode'
and variable_name!=    'session_track_system_variables';""")
        data=cursor.fetchall()
        pprint.pprint(jsonify(data))
        return jsonify(data)
    finally:
        if 'mydb' in locals():
            mydb.close()


if __name__ == "__main__":
    app.run(debug=True,port=12345)
