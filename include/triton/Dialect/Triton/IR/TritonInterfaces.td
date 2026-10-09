#ifndef TRITON_INTERFACES
#define TRITON_INTERFACES

include "mlir/IR/OpBase.td"

def TensorSizeTrait : NativeOpTrait<"TensorSizeTrait">;
def VerifyTensorLayoutsTrait : NativeOpTrait<"VerifyTensorLayoutsTrait">;
def DotLike : NativeOpTrait<"DotLike">;
def SameOperandsEncoding : NativeOpTrait<"SameOperandsEncoding">;
def SameOperandsAndResultEncoding : NativeOpTrait<"SameOperandsAndResultEncoding">;
def SameLoadStoreOperandsShape : NativeOpTrait<"SameLoadStoreOperandsShape">;
def SameLoadStoreOperandsAndResultShape : NativeOpTrait<"SameLoadStoreOperandsAndResultShape">;
def SameLoadStoreOperandsEncoding : NativeOpTrait<"SameLoadStoreOperandsEncoding">;
def SameLoadStoreOperandsAndResultEncoding : NativeOpTrait<"SameLoadStoreOperandsAndResultEncoding">;

#endif // TRITON_INTERFACES

def MemDescEncodingTrait : AttrInterface<"MemDescEncodingTrait"> {
    let cppNamespace = "::mlir::triton";
    let methods = [
        InterfaceMethod<"Verify MemDesc encoding",
                        "mlir::LogicalResult", "verifyMemDescEncoding",
                        (ins "llvm::ArrayRef<int64_t>":$shape, "mlir::Type":$elementType, "llvm::function_ref<mlir::InFlightDiagnostic()>":$emitError)
        >
    ];
}
