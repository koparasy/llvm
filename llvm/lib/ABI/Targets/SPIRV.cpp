//===- SPIRV.cpp - SPIR-V ABI Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/DefaultTargetInfo.h"
#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/IR/CallingConv.h"

namespace llvm::abi {

// Mirrors Clang's SPIRVABIInfo: the default classification applies, except
// for SPIR_KERNEL arguments.
class SPIRVTargetInfo : public DefaultTargetInfo {
private:
  ABICompatInfo CompatInfo;

  // When set, SPIR_KERNEL aggregate arguments are passed indirectly (byval)
  // so the callee gets its own copy. The consumer sets this for device
  // compilations, matching classic CodeGen's isTargetDevice() gate.
  bool KernelPassAggregatesIndirect;

  // Address space of the pointer an aggregate is returned through (sret). It
  // follows the language's default address space: generic for SYCL, private
  // for OpenCL C, matching classic CodeGen.
  unsigned SRetAddrSpace;

  ArgInfo classifyReturnType(const Type *RetTy) const {
    if (!RetTy->isVoid() && isAggregateTypeForABI(RetTy))
      return getNaturalAlignIndirect(RetTy, SRetAddrSpace);
    return DefaultTargetInfo::classifyReturnType(RetTy);
  }

  ArgInfo classifyKernelArgumentType(const Type *ArgTy) const {
    // A bool is not a legal SPIR-V kernel argument; pass it in the host's
    // in-memory representation (i8), as classic CodeGen does for SYCL.
    if (KernelPassAggregatesIndirect)
      if (const auto *IT = dyn_cast<IntegerType>(ArgTy); IT && IT->isBool())
        return ArgInfo::getDirect(
            TB.getIntegerType(8, Align(1), /*Signed=*/false));
    // TODO: Clang also coerces default address space pointer arguments to
    // CrossWorkGroup pointers, which needs the target's language address space
    // mapping.

    // Force copying aggregate type in kernel arguments by value. This is
    // required for the object copied to be valid on the device. TODO:
    // hardcoding to 0 should be revisited if HIPSPV / byval starts making use
    // of the AS of an indirect arg.
    if (KernelPassAggregatesIndirect && isAggregateTypeForABI(ArgTy))
      return getNaturalAlignIndirect(ArgTy, /*AddrSpace=*/0, /*ByVal=*/true);

    return classifyArgumentType(ArgTy);
  }

public:
  SPIRVTargetInfo(TypeBuilder &Builder, const ABICompatInfo &Info,
                  bool KernelPassAggregatesIndirect, unsigned SRetAddrSpace)
      : DefaultTargetInfo(Builder), CompatInfo(Info),
        KernelPassAggregatesIndirect(KernelPassAggregatesIndirect),
        SRetAddrSpace(SRetAddrSpace) {}

  const ABICompatInfo &getABICompatInfo() const override { return CompatInfo; }

  void computeInfo(FunctionInfo &FI) const override {
    // The logic is the same as in DefaultTargetInfo, except for kernel
    // arguments.
    if (!maybeCommonClassifyReturnType(FI))
      FI.getReturnInfo() = classifyReturnType(FI.getReturnType());

    bool IsKernel = FI.getCallingConvention() == CallingConv::SPIR_KERNEL;
    for (auto &I : FI.arguments())
      I.Info = IsKernel ? classifyKernelArgumentType(I.ABIType)
                        : classifyArgumentType(I.ABIType);
  }
};

std::unique_ptr<TargetInfo>
createSPIRVTargetInfo(TypeBuilder &TB, const ABICompatInfo &Compat,
                      bool KernelPassAggregatesIndirect,
                      unsigned SRetAddrSpace) {
  return std::make_unique<SPIRVTargetInfo>(
      TB, Compat, KernelPassAggregatesIndirect, SRetAddrSpace);
}

} // namespace llvm::abi
