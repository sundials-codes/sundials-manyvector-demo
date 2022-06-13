/*---------------------------------------------------------------
  Programmer(s): Daniel R. Reynolds @ SMU
  ----------------------------------------------------------------
  Copyright (c) 2022, Southern Methodist University.
  All rights reserved.
  For details, see the LICENSE file.
  ----------------------------------------------------------------
  Header file for SUNMemory "mirror" extension for allocating and
  managing pairs of host/device data.
  ---------------------------------------------------------------*/

// Only include this file once (if included multiple times).
#ifndef __SUNMEMORY_MIRROR_HPP__
#define __SUNMEMORY_MIRROR_HPP__

// Relevant SUNDIALS includes.
#include <sundials/sundials_memory.h>

template<typename T>
class SUNMemoryMirror {

public:

  // Constructor: allocates both host and device objects of type 'T', using the
  //   'helper' SUNMemoryHelper and 'queue' for device allocation.
  SUNMemoryMirror(SUNMemoryHelper helper, void* queue) :
    helper_(helper), queue_(queue), h_mem_(nullptr), d_mem_(nullptr) {
    SUNMemoryHelper_Alloc(helper, &h_mem_, sizeof(T), SUNMEMTYPE_HOST, nullptr);
    SUNMemoryHelper_Alloc(helper, &d_mem_, sizeof(T), SUNMEMTYPE_DEVICE, queue);
  }

  // Empty Constructor: only used for exception handling
  SUNMemoryMirror() :
    helper_(nullptr), queue_(nullptr), h_mem_(nullptr), d_mem_(nullptr) { }

  // Destructor: frees host and device arrays.
  ~SUNMemoryMirror() {
    SUNMemoryHelper_Dealloc(helper_, h_mem_, nullptr);
    SUNMemoryHelper_Dealloc(helper_, d_mem_, queue_ );
  }

  // Utility routines to copy host <-> device data.
  int HostToDevice() { return SUNMemoryHelper_Copy(helper_, h_mem_, d_mem_, sizeof(T), queue_); }
  int DeviceToHost() { return SUNMemoryHelper_Copy(helper_, d_mem_, h_mem_, sizeof(T), queue_); }

  // Accessor functions to return the host and device data pointers.
  T* HPtr() { return static_cast<T*>(h_mem_->ptr); }
  T* DPtr() { return static_cast<T*>(d_mem_->ptr); }

  // Query routine to check for valid SUNMemoryMirror object
  bool IsValid() { return ((h_mem_ != nullptr) && (d_mem_ != nullptr)); }

private:

  SUNMemory h_mem_;
  SUNMemory d_mem_;
  void* queue_;
  SUNMemoryHelper helper_;

};

#endif
