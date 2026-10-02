//===- SPIRVTargetInfoTest.cpp - SPIR-V ABI classification tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "gtest/gtest.h"

namespace {

using namespace llvm;

using ABIType = llvm::abi::Type;
using llvm::abi::ABICompatInfo;
using llvm::abi::ArgInfo;
using llvm::abi::createSPIRVTargetInfo;
using llvm::abi::FieldInfo;
using llvm::abi::FunctionInfo;
using llvm::abi::RecordFlags;
using llvm::abi::StructPacking;
using llvm::abi::TargetInfo;
using llvm::abi::TypeBuilder;

class SPIRVTargetInfoTest : public ::testing::Test {
protected:
  llvm::BumpPtrAllocator Alloc;
  TypeBuilder TB;
  ABICompatInfo Compat;
  const ABIType *Bool;
  const ABIType *I32;
  const ABIType *F32;
  const ABIType *Void;
  /// A small register-passable record { i32, float }.
  const ABIType *Pair;

  SPIRVTargetInfoTest()
      : TB(Alloc),
        Bool(TB.getIntegerType(1, llvm::Align(1), /*Signed=*/false)),
        I32(TB.getIntegerType(32, llvm::Align(4), /*Signed=*/true)),
        F32(TB.getFloatType(llvm::APFloat::IEEEsingle(), llvm::Align(4))),
        Void(TB.getVoidType()),
        Pair(TB.getRecordType({FieldInfo(I32, 0), FieldInfo(F32, 32)},
                              llvm::TypeSize::getFixed(64), llvm::Align(4),
                              llvm::Align(4), StructPacking::Default, {}, {},
                              RecordFlags::CanPassInRegisters)) {}

  std::unique_ptr<TargetInfo> target(bool KernelPassAggregatesIndirect = true,
                                     unsigned SRetAddrSpace = 0) {
    return createSPIRVTargetInfo(TB, Compat, KernelPassAggregatesIndirect,
                                 SRetAddrSpace);
  }

  /// The classification of the single argument \p ArgTy under \p CC.
  ArgInfo classifyArg(const ABIType *ArgTy, CallingConv::ID CC,
                      const TargetInfo &TI) {
    std::unique_ptr<FunctionInfo> FI = FunctionInfo::create(CC, Void, {ArgTy});
    TI.computeInfo(*FI);
    return FI->getArgInfo(0).Info;
  }

  /// The return classification of \p RetTy under \p CC.
  ArgInfo classifyRet(const ABIType *RetTy, CallingConv::ID CC,
                      const TargetInfo &TI) {
    std::unique_ptr<FunctionInfo> FI = FunctionInfo::create(CC, RetTy, {});
    TI.computeInfo(*FI);
    return FI->getReturnInfo();
  }
};

TEST_F(SPIRVTargetInfoTest, KernelAggregateIsByVal) {
  std::unique_ptr<TargetInfo> TI = target();
  ArgInfo Info = classifyArg(Pair, CallingConv::SPIR_KERNEL, *TI);
  ASSERT_TRUE(Info.isIndirect());
  EXPECT_TRUE(Info.getIndirectByVal());
  EXPECT_EQ(Info.getIndirectAlign(), llvm::Align(4));
  EXPECT_EQ(Info.getIndirectAddrSpace(), 0u);
}

TEST_F(SPIRVTargetInfoTest, KernelAggregateWithoutFlagUsesDefault) {
  std::unique_ptr<TargetInfo> TI = target(/*KernelPassAggregatesIndirect=*/false);
  ArgInfo Kernel = classifyArg(Pair, CallingConv::SPIR_KERNEL, *TI);
  ArgInfo Func = classifyArg(Pair, CallingConv::SPIR_FUNC, *TI);
  EXPECT_EQ(Kernel.isIndirect(), Func.isIndirect());
  EXPECT_EQ(Kernel.isDirect(), Func.isDirect());
}

TEST_F(SPIRVTargetInfoTest, NonKernelAggregateIsByVal) {
  std::unique_ptr<TargetInfo> TI = target();
  ArgInfo Info = classifyArg(Pair, CallingConv::SPIR_FUNC, *TI);
  ASSERT_TRUE(Info.isIndirect());
  EXPECT_TRUE(Info.getIndirectByVal());
}

TEST_F(SPIRVTargetInfoTest, KernelScalarIsDirect) {
  std::unique_ptr<TargetInfo> TI = target();
  ArgInfo Info = classifyArg(I32, CallingConv::SPIR_KERNEL, *TI);
  EXPECT_TRUE(Info.isDirect());
}

TEST_F(SPIRVTargetInfoTest, KernelBoolIsI8) {
  std::unique_ptr<TargetInfo> TI = target();
  ArgInfo Info = classifyArg(Bool, CallingConv::SPIR_KERNEL, *TI);
  ASSERT_TRUE(Info.isDirect());
  const auto *IT =
      llvm::dyn_cast_or_null<llvm::abi::IntegerType>(Info.getCoerceToType());
  ASSERT_NE(IT, nullptr);
  EXPECT_EQ(IT->getSizeInBits().getFixedValue(), 8u);
}

TEST_F(SPIRVTargetInfoTest, KernelBoolWithoutFlagIsExtended) {
  std::unique_ptr<TargetInfo> TI = target(/*KernelPassAggregatesIndirect=*/false);
  ArgInfo Info = classifyArg(Bool, CallingConv::SPIR_KERNEL, *TI);
  EXPECT_TRUE(Info.isExtend());
}

TEST_F(SPIRVTargetInfoTest, AggregateReturnUsesSRetAddrSpace) {
  std::unique_ptr<TargetInfo> Default = target();
  ArgInfo Info = classifyRet(Pair, CallingConv::SPIR_FUNC, *Default);
  ASSERT_TRUE(Info.isIndirect());
  EXPECT_EQ(Info.getIndirectAddrSpace(), 0u);

  std::unique_ptr<TargetInfo> Generic =
      target(/*KernelPassAggregatesIndirect=*/true, /*SRetAddrSpace=*/4);
  Info = classifyRet(Pair, CallingConv::SPIR_FUNC, *Generic);
  ASSERT_TRUE(Info.isIndirect());
  EXPECT_EQ(Info.getIndirectAddrSpace(), 4u);
}

TEST_F(SPIRVTargetInfoTest, VoidReturnIsIgnored) {
  std::unique_ptr<TargetInfo> TI = target();
  EXPECT_TRUE(classifyRet(Void, CallingConv::SPIR_FUNC, *TI).isIgnore());
}

} // namespace
