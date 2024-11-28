import argparse
import time
from flask import Flask, request
import mysql

from autotune.database.mysqldb import MysqlDB
from autotune.database.postgresqldb import PostgresqlDB
from autotune.dbenv import DBEnv
from autotune.tuner import DBTuner
from autotune.utils.config import parse_args

app = Flask(__name__)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="scripts/demo.ini", help="config file"
)
parser.add_argument("run")
parser.add_argument("--no-debugger", action="store_true")
parser.add_argument("--no-reload", action="store_true")
opt = parser.parse_args()


args_db, args_tune = parse_args(opt.config)
if args_db["db"] == "mysql":
    db = MysqlDB(args_db)
elif args_db["db"] == "postgresql":
    db = PostgresqlDB(args_db)

env = DBEnv(args_db, args_tune, db)
tuner = DBTuner(args_db, args_tune, env)


mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",
    database="voter",
)


@app.post("/tune")
def tune():
    queries = request.get_json()
    for i, query in enumerate(queries):
        with open(
            f"/root/OpAdviserPrivate/autotune/demo_query/queries-mysql-new/{i}.sql", "w"
        ) as f:
            f.write(
                f"""select current_timestamp(6) into @query_start;
set @query_name='{i}';
{query}
set @query_time_ms= timestampdiff(microsecond, @query_start, current_timestamp(6))/1000;
SELECT @query_name, @query_time_ms;
"""
            )
    tuner.tune()


@app.post("/query")
def query():
    query = request.get_json()
    cursor = mydb.cursor()
    start_time = time.time()
    cursor.execute(query)
    end_time = time.time()
    return {
        "results": cursor.fetchall(),
        "query_time": end_time - start_time,
    }


@app.post("/knobs")
def knobs():
    cursor = mydb.cursor()
    cursor.execute("SHOW VARIABLES;")
    return {
        "results": cursor.fetchall(),
    }


if __name__ == "__main__":
    app.run(debug=True)
