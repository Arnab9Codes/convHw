
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I..
EXE	        = conv2d
OBJ	        = main.o support.o

default: $(EXE)


main.o: main.cu kernel.cu support.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)

