#!/bin/bash

echo "Hello, $0! "

EXE=/home/yaoyu/Yuanwei/LineMatching/cmake-build-release/test_optimization

DATA_DIR=/home/yaoyu/Yuanwei/LineMatching/data
TEMPLATE_NAME_PREFIX="0839-0001-"

# TEST_NAME_PREFIX="0839-0002-"
# OUT_NAME_PREFIX="0002-"

TEST_NAME_PREFIX="0852-0056-"
OUT_NAME_PREFIX="0056-"

# CASE=05
# $EXE \
# 	${DATA_DIR}/${TEMPLATE_NAME_PREFIX}${CASE}.jpg \
# 	${DATA_DIR}/${TEST_NAME_PREFIX}${CASE}.jpg \
# 	-name=${OUT_NAME_PREFIX}${CASE} \
# 	-resize=-1 \
# 	-filter_length=10 \
# 	-weight_sigma=-1 \

CASE=07
$EXE \
	${DATA_DIR}/${TEMPLATE_NAME_PREFIX}${CASE}.jpg \
	${DATA_DIR}/${TEST_NAME_PREFIX}${CASE}.jpg \
	-name=${OUT_NAME_PREFIX}${CASE} \
	-resize=-1 \
	-filter_length=20 \
	-weight_sigma=-1 \

# CASE=08
# $EXE \
# 	${DATA_DIR}/${TEMPLATE_NAME_PREFIX}${CASE}.jpg \
# 	${DATA_DIR}/${TEST_NAME_PREFIX}${CASE}.jpg \
# 	-name=${OUT_NAME_PREFIX}${CASE} \
# 	-resize=1024 \
# 	-filter_length=20 \
# 	-weight_sigma=100 \
# 	# -binarise_threshold=10 \

# CASE=09
# $EXE \
# 	${DATA_DIR}/${TEMPLATE_NAME_PREFIX}${CASE}.jpg \
# 	${DATA_DIR}/${TEST_NAME_PREFIX}${CASE}.jpg \
# 	-name=${OUT_NAME_PREFIX}${CASE} \
# 	-resize=-1 \
# 	-filter_length=20 \
# 	-weight_sigma=-1 \
# 	# -binarise_threshold=10 \

# CASE=13
# $EXE \
# 	${DATA_DIR}/${TEMPLATE_NAME_PREFIX}${CASE}.jpg \
# 	${DATA_DIR}/${TEST_NAME_PREFIX}${CASE}.jpg \
# 	-name=${OUT_NAME_PREFIX}${CASE} \
# 	-resize=1024 \
# 	-filter_length=50 \
# 	-weight_sigma=500 \
# 	-binarise_threshold=10 \

# CASE=14
# $EXE \
# 	${DATA_DIR}/${TEMPLATE_NAME_PREFIX}${CASE}.jpg \
# 	${DATA_DIR}/${TEST_NAME_PREFIX}${CASE}.jpg \
# 	-name=${OUT_NAME_PREFIX}${CASE} \
# 	-resize=1024 \
# 	-filter_length=10 \
# 	-weight_sigma=500 \
# 	-binarise_threshold=10\

# CASE=16
# $EXE \
# 	${DATA_DIR}/${TEMPLATE_NAME_PREFIX}${CASE}.jpg \
# 	${DATA_DIR}/${TEST_NAME_PREFIX}${CASE}.jpg \
# 	-name=${OUT_NAME_PREFIX}${CASE} \
# 	-resize=1024 \
# 	-filter_length=100 \
# 	-weight_sigma=500 \
# 	-binarise_threshold=10 \

echo "$0 done. "
