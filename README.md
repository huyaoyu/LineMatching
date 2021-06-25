# Line matching

## Docker
The project requires Ubuntu 18.04, OpenCV 4.0.0, and Ceres Solver 2.0. I have a set of docker files for building an image for testing this project.

Download the docker files from [here][docker_and_data] (PIN: cl09). Perform incremental build by using `ubuntu18.dockerfile`, `ceres.dockerfile`, and `opencv400.dockerfile` in order. 
- For `ubuntu18.dockerfile`, please set the `user_name`, `user_id`, `group_name`, and `group_id` properly when doing `docker build`.
- For `ceres.dockerfile` and `opencv400.dockerfile`, remember to set the args for the `FROM` command.

[docker_and_data]: https://pan.baidu.com/s/14q_j89uGU4DG4QKkV6LuYA

## Build the project
This project is an ordinary CMake project. Use `-DCMAKE_BUILD_TYPE=Release` to build the release version.

## Simple test
A sample test script is saved at `/run/run.sh`. Please copy it to a new location and modify the settings to reflect the correct sample data setting.
A set of sample data could also be found at the location where the docker files are ([here][docker_and_data]).
