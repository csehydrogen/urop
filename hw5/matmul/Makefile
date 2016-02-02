TARGET=mat_mul
OBJS=mat_mul.o timers.o mat_mul_opencl.o
LIBS=-lOpenCL

CC=gcc
CFLAGS=-std=c99 -g -O2 -Wall
LDFLAGS=

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	rm -rf $(TARGET) $(OBJS) task*

run: $(TARGET)
	thorq --add --mode single --device gpu ./$(TARGET)
