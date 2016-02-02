CXX=g++
CXXFLAGS=-O2 -Wall
LDLIBS=-lOpenCL -lrt -lstdc++

all: kmeans_seq kmeans_opencl

kmeans_seq: kmeans_seq.o kmeans_main.o

kmeans_opencl: kmeans_opencl.o kmeans_main.o

run_seq:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 65536 data.point 64
	thorq --add kmeans_seq centroid.point data.point result_seq.class final_centroid_seq.point 1024

run_opencl:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 65536 data.point 64
	thorq --add --mode single --device gpu kmeans_opencl centroid.point data.point result_opencl.class final_centroid_opencl.point 1024

run: run_opencl

image:
	./plot_data.py input centroid.point data.point input.png
	./plot_data.py result final_centroid_opencl.point data.point result_opencl.class result.png

clean:
	rm -f kmeans_seq kmeans_opencl *.o *.point *.class task_* *.png
