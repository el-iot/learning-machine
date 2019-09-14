#!/Users/el/myenv/bin/python

from threading import Timer
from webbrowser import open_new_tab

import pandas
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy

PORT = 2000
OPEN = False

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("index.html")


if __name__ == "__main__":
    if OPEN:
        open_new_tab(f"http://localhost:{PORT}/")
    Timer(2, app.run(debug=True, port=PORT))
