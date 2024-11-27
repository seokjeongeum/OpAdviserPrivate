from flask import Flask, request

from scripts import optimize

app = Flask(__name__)


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
    optimize.f()


app.run(debug=True)
