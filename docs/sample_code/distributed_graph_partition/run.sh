#!/bin/bash
# Launch scheduler and worker for distributed graph partition.
execute_path=$(pwd)
self_path=$(dirname $0)

# Set public environment.
export MS_WORKER_NUM=8
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8118
export DATA_PATH=$1

# Launch scheduler.
export MS_ROLE=MS_SCHED
rm -rf ${execute_path}/sched/
mkdir ${execute_path}/sched/
cd ${execute_path}/sched/ || exit
python ${self_path}/../train.py > sched.log 2>&1 &
sched_pid=`echo $!`


# Launch workers.
export MS_ROLE=MS_WORKER
worker_pids=()
for((i=0;i<$MS_WORKER_NUM;i++));
do
  rm -rf ${execute_path}/worker_$i/
  mkdir ${execute_path}/worker_$i/
  cd ${execute_path}/worker_$i/ || exit
  python ${self_path}/../train.py > worker_$i.log 2>&1 &
  worker_pids[${i}]=`echo $!`
done

# Wait for workers to exit.
for((i=0; i<${MS_WORKER_NUM}; i++)); do
  wait ${worker_pids[i]}
  status=`echo $?`
  if [ "${status}" != "0" ]; then
      echo "[ERROR] train failed. Failed to wait worker_{$i}, status: ${status}"
      exit 1
  fi
done

# Wait for scheduler to exit.
if [ "${status}" != "0" ]; then
  wait ${sched_pid}
  status=`echo $?`
  if [ "${status}" != "0" ]; then
    echo "[ERROR] train failed. Failed to wait scheduler, status: ${status}"
    exit 1
  fi
fi

exit 0