#!/bin/bash

username="lishiwei"
count=$(pgrep -u $username -c python)
echo "number of python process of lishiwei: $count"

ps -ef | grep python | grep lishiwei |grep server.py