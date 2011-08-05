MODELNAME := _ConvNet

INCLUDES :=  -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -I./include
LIB := -lpthread -L$(ATLAS_LIB_PATH) -L$(CUDA_INSTALL_PATH)/lib64 -lcblas

USECUBLAS   := 1

PYTHON_VERSION=$(shell python -V 2>&1 | cut -d ' ' -f 2 | cut -d '.' -f 1,2)
LIB += -lpython$(PYTHON_VERSION)

GENCODE_ARCH := -gencode=arch=compute_20,code=\"sm_20,compute_20\"
COMMONFLAGS := -DNUMPY_INTERFACE -DMODELNAME=$(MODELNAME) -DINITNAME=init$(MODELNAME)

EXECUTABLE	:= $(MODELNAME).so

CUFILES				:= $(shell echo src/*.cu)
CU_DEPS				:= $(shell echo include/*.cuh )
CCFILES				:= $(shell echo src/*.cpp)
C_DEPS				:= $(shell echo include/*.h)

include common-gcc-cuda-4.0.mk
	
makedirectories:
	$(VERBOSE)mkdir -p $(LIBDIR)
	$(VERBOSE)mkdir -p $(OBJDIR)/src/kernel
	$(VERBOSE)mkdir -p $(TARGETDIR)
