# Makefile

CC = nvcc
CXX = g++
CXXFLAGS = -O3 -std=c++11
CUDAFLAGS = -arch=sm_35
INCLUDES = -Iinclude -Isrc
SRC = $(wildcard src/**/*.cu)
OBJ = $(SRC:.cu=.o)
TARGET = transformer

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CUDAFLAGS) $(OBJ) -o $@

%.o: %.cu
	$(CC) $(CUDAFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean