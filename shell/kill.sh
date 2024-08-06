#!/bin/bash

PROCESS=`ps -ef | grep python | grep lishiwei | awk '{print $2}' | xargs kill -9`
