#!/usr/bin/env python3
import os

import flask

from buo_2017 import create_app

server = flask.Flask('app')
server.secret_key = os.environ.get('secret_key', 'secret')
create_app(server=server, url_base_pathname='/buo-2017/')


@server.route('/')
def homepage():
    content = """
    <h1>TIKS Stats</h1>
    You can find some stats for the team here
    <ul>
    <li><a href="buo-2017">BUO 2017</a></li>
    </ul>
    """
    return content


@server.route('/robots.txt')
def static_from_root():
    return flask.send_from_directory(
        server.static_folder, flask.request.path[1:]
    )


if __name__ == '__main__':
    server.run(debug=True)
