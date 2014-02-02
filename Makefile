SHELL = sh -xv

MODELNAME := _ConvNet

INCLUDES :=  -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -I./include -I./include/common -I./include/cudaconv2 -I./include/nvmatrix -I./include/dropc/ -I./include/tdata/
LIB := -L/usr/lib/nvidia-310-updates/ -lpthread -L$(ATLAS_LIB_PATH) -L$(CUDA_INSTALL_PATH)/lib64 -lcblas -lm -lpthread -lX11 -fopenmp

USECUBLAS   := 1

PYTHON_VERSION=$(shell python -V 2>&1 | cut -d ' ' -f 2 | cut -d '.' -f 1,2)
LIB += -lpython$(PYTHON_VERSION)

GENCODE_ARCH := -gencode=arch=compute_20,code=\"sm_20,compute_20\" -gencode=arch=compute_30,code=\"sm_30,compute_30\"

COMMONFLAGS := -DNUMPY_INTERFACE -DMODELNAME=$(MODELNAME) -DINITNAME=init$(MODELNAME)

EXECUTABLE	:= $(MODELNAME).so

CUFILES				:= $(shell echo src/*.cu src/cudaconv2/*.cu src/nvmatrix/*.cu src/dropc/*.cu)
CU_DEPS				:= $(shell echo include/*.cuh include/cudaconv2/*.cuh include/nvmatrix/*.cuh)
CCFILES				:= $(shell echo src/common/*.cpp src/tdata/*.cpp)
C_DEPS				:= $(shell echo include/common/*.h)

include common-gcc-cuda-5.0.mk
	
makedirectories:
	$(VERBOSE)mkdir -p $(LIBDIR)
	$(VERBOSE)mkdir -p $(OBJDIR)/src/cudaconv2
	$(VERBOSE)mkdir -p $(OBJDIR)/src/nvmatrix
	$(VERBOSE)mkdir -p $(OBJDIR)/src/common
	$(VERBOSE)mkdir -p $(OBJDIR)/src/dropc
	$(VERBOSE)mkdir -p $(OBJDIR)/src/tdata
	$(VERBOSE)mkdir -p $(TARGETDIR)
