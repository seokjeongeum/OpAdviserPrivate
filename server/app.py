import argparse
from flask import Flask, request

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


@app.post("/")
def hello_world():
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


app.run(debug=True)
