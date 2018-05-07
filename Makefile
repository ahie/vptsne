TARGET=./build
TF_CFLAGS=$(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CUDA_LIB?=/usr/local/cuda/lib64

.PHONY: all compile clean

all: compile

compile: cu cc

cu:
	mkdir -p $(TARGET)
	nvcc -std=c++11 -c -o $(TARGET)/tsne_loss.cu.o src/tsne_loss.cu \
	$(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

cc:
	g++ -std=c++11 -shared -o vptsne/tsne_loss.so src/tsne_loss.cc \
	$(TARGET)/tsne_loss.cu.o $(TF_CFLAGS) -fPIC -lcudart -lcurand $(TF_LFLAGS) -L$(CUDA_LIB) -D GOOGLE_CUDA=1

clean:
	rm -vfr $(TARGET)

