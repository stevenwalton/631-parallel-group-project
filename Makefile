CC=/usr/bin/g++
CFLAGS=-Wall -std=c++11

SRC=$(wildcard *.cpp)
OBJS=$(wildcard *.o)
EXEC=serial_nn.cpp

serial: ${EXEC}
	${CC} -o $@ ${EXEC} ${CFLAGS}
