NVCC = nvcc
NVCCFLAGS = -arch=sm_86

TARGET = main
SRC = gemm.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^
