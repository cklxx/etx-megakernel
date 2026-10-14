; ModuleID = '/root/sess/20261014T183414Z-68d6f6/qwen/b_kept_l1/imp/layernorm_kernels.hip'
source_filename = "/root/sess/20261014T183414Z-68d6f6/qwen/b_kept_l1/imp/layernorm_kernels.hip"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%"struct.rocprim::ROCPRIM_0_NS::detail::raw_storage" = type { [64 x i8] }

$_ZN4vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi2ELb1EEEvPT_PKS3_lllllS6_lfii = comdat any

$_ZZN4vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi2ELb1EEEvPT_PKS3_lllllS6_lfiiE10s_variance = comdat any

$_ZZN4vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi2ELb1EEEvPT_PKS3_lllllS6_lfiiE11reduceStore = comdat any

@_ZZN4vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi2ELb1EEEvPT_PKS3_lllllS6_lfiiE10s_variance = linkonce_odr hidden local_unnamed_addr addrspace(3) global float undef, comdat, align 4
@_ZZN4vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi2ELb1EEEvPT_PKS3_lllllS6_lfiiE11reduceStore = linkonce_odr hidden local_unnamed_addr addrspace(3) global %"struct.rocprim::ROCPRIM_0_NS::detail::raw_storage" undef, comdat, align 8
@__hip_cuid_322149f5921a02eb = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_322149f5921a02eb to ptr)], section "llvm.metadata"

; Function Attrs: convergent mustprogress nofree norecurse nounwind
define protected amdgpu_kernel void @_ZN4vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi2ELb1EEEvPT_PKS3_lllllS6_lfii(ptr addrspace(1) noalias noundef writeonly captures(none) %0, ptr addrspace(1) noalias noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i64 noundef %6, ptr addrspace(1) noalias noundef readonly captures(none) %7, i64 noundef %8, float noundef %9, i32 noundef %10, i32 noundef %11) local_unnamed_addr #0 comdat {
  %13 = tail call i64 @__ockl_get_group_id(i32 noundef 0) #7
  %14 = and i64 %13, 4294967295
  %15 = mul nsw i64 %14, %2
  %16 = getelementptr inbounds [2 x i8], ptr addrspace(1) %1, i64 %15
  %17 = addrspacecast ptr addrspace(1) %16 to ptr
  %18 = mul nsw i64 %14, %8
  %19 = getelementptr inbounds [2 x i8], ptr addrspace(1) %7, i64 %18
  %20 = tail call i64 @__ockl_get_local_id(i32 noundef 0) #7
  %21 = trunc i64 %20 to i32
  %22 = tail call i64 @__ockl_get_local_size(i32 noundef 0) #7
  %23 = trunc i64 %22 to i32
  %24 = ptrtoint ptr %17 to i64
  %25 = and i64 %24, 15
  %26 = icmp eq i64 %25, 0
  %27 = and i32 %11, 7
  %28 = icmp eq i32 %27, 0
  %29 = and i1 %28, %26
  br i1 %29, label %30, label %61

30:                                               ; preds = %12
  %31 = ashr exact i32 %11, 3
  %32 = icmp sgt i32 %31, %21
  br i1 %32, label %33, label %134

33:                                               ; preds = %30, %33
  %34 = phi float [ %58, %33 ], [ 0.000000e+00, %30 ]
  %35 = phi i32 [ %59, %33 ], [ %21, %30 ]
  %36 = sext i32 %35 to i64
  %37 = getelementptr inbounds [16 x i8], ptr addrspace(1) %16, i64 %36
  %38 = load <8 x i16>, ptr addrspace(1) %37, align 16
  %39 = zext <8 x i16> %38 to <8 x i32>
  %40 = shl nuw <8 x i32> %39, splat (i32 16)
  %41 = bitcast <8 x i32> %40 to <8 x float>
  %42 = fmul contract <8 x float> %41, %41
  %43 = extractelement <8 x float> %42, i64 0
  %44 = fadd contract float %34, %43
  %45 = extractelement <8 x float> %42, i64 1
  %46 = fadd contract float %45, %44
  %47 = extractelement <8 x float> %42, i64 2
  %48 = fadd contract float %47, %46
  %49 = extractelement <8 x float> %42, i64 3
  %50 = fadd contract float %49, %48
  %51 = extractelement <8 x float> %42, i64 4
  %52 = fadd contract float %51, %50
  %53 = extractelement <8 x float> %42, i64 5
  %54 = fadd contract float %53, %52
  %55 = extractelement <8 x float> %42, i64 6
  %56 = fadd contract float %55, %54
  %57 = extractelement <8 x float> %42, i64 7
  %58 = fadd contract float %57, %56
  %59 = add nsw i32 %35, %23
  %60 = icmp slt i32 %59, %31
  br i1 %60, label %33, label %134, !llvm.loop !8

61:                                               ; preds = %12
  %62 = trunc i64 %24 to i32
  %63 = sub i32 0, %62
  %64 = lshr i32 %63, 1
  %65 = and i32 %64, 7
  %66 = tail call noundef range(i32 -2147483648, 8) i32 @llvm.smin.i32(i32 range(i32 0, 8) %65, i32 %11)
  %67 = icmp sgt i32 %66, %21
  br i1 %67, label %75, label %68

68:                                               ; preds = %75, %61
  %69 = phi float [ 0.000000e+00, %61 ], [ %85, %75 ]
  %70 = sext i32 %66 to i64
  %71 = getelementptr inbounds [2 x i8], ptr addrspace(1) %16, i64 %70
  %72 = sub nsw i32 %11, %66
  %73 = lshr i32 %72, 3
  %74 = icmp sgt i32 %73, %21
  br i1 %74, label %93, label %88

75:                                               ; preds = %61, %75
  %76 = phi float [ %85, %75 ], [ 0.000000e+00, %61 ]
  %77 = phi i32 [ %86, %75 ], [ %21, %61 ]
  %78 = sext i32 %77 to i64
  %79 = getelementptr inbounds [2 x i8], ptr addrspace(1) %16, i64 %78
  %80 = load i16, ptr addrspace(1) %79, align 2, !tbaa !10
  %81 = zext i16 %80 to i32
  %82 = shl nuw i32 %81, 16
  %83 = bitcast i32 %82 to float
  %84 = fmul contract float %83, %83
  %85 = fadd contract float %76, %84
  %86 = add nsw i32 %77, %23
  %87 = icmp slt i32 %86, %66
  br i1 %87, label %75, label %68, !llvm.loop !13

88:                                               ; preds = %93, %68
  %89 = phi float [ %69, %68 ], [ %118, %93 ]
  %90 = and i32 %72, -8
  %91 = add nsw i32 %90, %21
  %92 = icmp slt i32 %91, %72
  br i1 %92, label %121, label %134

93:                                               ; preds = %68, %93
  %94 = phi float [ %118, %93 ], [ %69, %68 ]
  %95 = phi i32 [ %119, %93 ], [ %21, %68 ]
  %96 = sext i32 %95 to i64
  %97 = getelementptr inbounds [16 x i8], ptr addrspace(1) %71, i64 %96
  %98 = load <8 x i16>, ptr addrspace(1) %97, align 2, !tbaa !10
  %99 = zext <8 x i16> %98 to <8 x i32>
  %100 = shl nuw <8 x i32> %99, splat (i32 16)
  %101 = bitcast <8 x i32> %100 to <8 x float>
  %102 = fmul contract <8 x float> %101, %101
  %103 = extractelement <8 x float> %102, i64 0
  %104 = fadd contract float %94, %103
  %105 = extractelement <8 x float> %102, i64 1
  %106 = fadd contract float %104, %105
  %107 = extractelement <8 x float> %102, i64 2
  %108 = fadd contract float %106, %107
  %109 = extractelement <8 x float> %102, i64 3
  %110 = fadd contract float %108, %109
  %111 = extractelement <8 x float> %102, i64 4
  %112 = fadd contract float %110, %111
  %113 = extractelement <8 x float> %102, i64 5
  %114 = fadd contract float %112, %113
  %115 = extractelement <8 x float> %102, i64 6
  %116 = fadd contract float %114, %115
  %117 = extractelement <8 x float> %102, i64 7
  %118 = fadd contract float %116, %117
  %119 = add nsw i32 %95, %23
  %120 = icmp slt i32 %119, %73
  br i1 %120, label %93, label %88, !llvm.loop !14

121:                                              ; preds = %88, %121
  %122 = phi float [ %131, %121 ], [ %89, %88 ]
  %123 = phi i32 [ %132, %121 ], [ %91, %88 ]
  %124 = sext i32 %123 to i64
  %125 = getelementptr inbounds [2 x i8], ptr addrspace(1) %71, i64 %124
  %126 = load i16, ptr addrspace(1) %125, align 2, !tbaa !10
  %127 = zext i16 %126 to i32
  %128 = shl nuw i32 %127, 16
  %129 = bitcast i32 %128 to float
  %130 = fmul contract float %129, %129
  %131 = fadd contract float %122, %130
  %132 = add nsw i32 %123, %23
  %133 = icmp slt i32 %132, %72
  br i1 %133, label %121, label %134, !llvm.loop !15

134:                                              ; preds = %121, %33, %30, %88
  %135 = phi float [ %58, %33 ], [ 0.000000e+00, %30 ], [ %89, %88 ], [ %131, %121 ]
  %136 = tail call i64 @__ockl_get_local_size(i32 noundef 0) #7
  %137 = trunc i64 %136 to i32
  %138 = tail call i64 @__ockl_get_local_id(i32 noundef 0) #7
  %139 = trunc i64 %138 to i32
  %140 = tail call range(i32 0, 33) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %141 = tail call noundef range(i32 0, 65) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %140)
  %142 = and i32 %139, -64
  %143 = tail call i32 @llvm.usub.sat.i32(i32 %137, i32 %142)
  %144 = and i32 %141, 63
  %145 = bitcast float %135 to i32
  %146 = add nuw nsw i32 %144, 1
  %147 = icmp ne i32 %144, 63
  %148 = zext i1 %147 to i32
  %149 = add nuw nsw i32 %141, %148
  %150 = shl nuw nsw i32 %149, 2
  %151 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %150, i32 %145)
  %152 = icmp ult i32 %146, %143
  %153 = bitcast i32 %151 to float
  %154 = fadd contract float %135, %153
  %155 = bitcast float %154 to i32
  %156 = select i1 %152, i32 %155, i32 %145
  %157 = select i1 %152, float %154, float %135
  %158 = add nuw nsw i32 %144, 2
  %159 = icmp samesign ult i32 %144, 62
  %160 = select i1 %159, i32 2, i32 0
  %161 = add nuw nsw i32 %160, %141
  %162 = shl nuw nsw i32 %161, 2
  %163 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %162, i32 %156)
  %164 = icmp ult i32 %158, %143
  %165 = bitcast i32 %163 to float
  %166 = fadd contract float %157, %165
  %167 = bitcast float %166 to i32
  %168 = select i1 %164, i32 %167, i32 %156
  %169 = select i1 %164, float %166, float %157
  %170 = add nuw nsw i32 %144, 4
  %171 = icmp samesign ult i32 %144, 60
  %172 = select i1 %171, i32 4, i32 0
  %173 = add nuw nsw i32 %172, %141
  %174 = shl nuw nsw i32 %173, 2
  %175 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %174, i32 %168)
  %176 = icmp ult i32 %170, %143
  %177 = bitcast i32 %175 to float
  %178 = fadd contract float %169, %177
  %179 = bitcast float %178 to i32
  %180 = select i1 %176, i32 %179, i32 %168
  %181 = select i1 %176, float %178, float %169
  %182 = add nuw nsw i32 %144, 8
  %183 = icmp samesign ult i32 %144, 56
  %184 = select i1 %183, i32 8, i32 0
  %185 = add nuw nsw i32 %184, %141
  %186 = shl nuw nsw i32 %185, 2
  %187 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %186, i32 %180)
  %188 = icmp ult i32 %182, %143
  %189 = bitcast i32 %187 to float
  %190 = fadd contract float %181, %189
  %191 = bitcast float %190 to i32
  %192 = select i1 %188, i32 %191, i32 %180
  %193 = select i1 %188, float %190, float %181
  %194 = add nuw nsw i32 %144, 16
  %195 = icmp samesign ult i32 %144, 48
  %196 = select i1 %195, i32 16, i32 0
  %197 = add nuw nsw i32 %196, %141
  %198 = shl nuw nsw i32 %197, 2
  %199 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %198, i32 %192)
  %200 = icmp ult i32 %194, %143
  %201 = bitcast i32 %199 to float
  %202 = fadd contract float %193, %201
  %203 = bitcast float %202 to i32
  %204 = select i1 %200, i32 %203, i32 %192
  %205 = add nuw nsw i32 %144, 32
  %206 = shl nuw nsw i32 %141, 2
  %207 = or i32 %206, 128
  %208 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %207, i32 %204)
  %209 = icmp ult i32 %205, %143
  %210 = select i1 %200, float %202, float %193
  %211 = bitcast i32 %208 to float
  %212 = fadd contract float %210, %211
  %213 = select i1 %209, float %212, float %210
  %214 = select i1 %152, float %213, float %135
  %215 = icmp eq i32 %141, 0
  br i1 %215, label %216, label %219

216:                                              ; preds = %134
  %217 = lshr i32 %139, 6
  %218 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @_ZZN4vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi2ELb1EEEvPT_PKS3_lllllS6_lfiiE11reduceStore, i32 %217
  store float %214, ptr addrspace(3) %218, align 4, !tbaa !16
  br label %219

219:                                              ; preds = %216, %134
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %220 = icmp ult i32 %139, 16
  br i1 %220, label %221, label %272

221:                                              ; preds = %219
  %222 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @_ZZN4vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi2ELb1EEEvPT_PKS3_lllllS6_lfiiE11reduceStore, i32 %141
  %223 = load float, ptr addrspace(3) %222, align 4, !tbaa !16
  %224 = add i32 %137, 63
  %225 = lshr i32 %224, 6
  %226 = and i32 %141, 15
  %227 = bitcast float %223 to i32
  %228 = add nuw nsw i32 %226, 1
  %229 = icmp ne i32 %226, 15
  %230 = zext i1 %229 to i32
  %231 = add nuw nsw i32 %141, %230
  %232 = shl nuw nsw i32 %231, 2
  %233 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %232, i32 %227)
  %234 = icmp samesign ult i32 %228, %225
  %235 = bitcast i32 %233 to float
  %236 = fadd contract float %223, %235
  %237 = bitcast float %236 to i32
  %238 = select i1 %234, i32 %237, i32 %227
  %239 = select i1 %234, float %236, float %223
  %240 = add nuw nsw i32 %226, 2
  %241 = icmp samesign ult i32 %226, 14
  %242 = select i1 %241, i32 2, i32 0
  %243 = add nuw nsw i32 %242, %141
  %244 = shl nuw nsw i32 %243, 2
  %245 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %244, i32 %238)
  %246 = icmp samesign ult i32 %240, %225
  %247 = bitcast i32 %245 to float
  %248 = fadd contract float %239, %247
  %249 = bitcast float %248 to i32
  %250 = select i1 %246, i32 %249, i32 %238
  %251 = select i1 %246, float %248, float %239
  %252 = add nuw nsw i32 %226, 4
  %253 = icmp samesign ult i32 %226, 12
  %254 = select i1 %253, i32 4, i32 0
  %255 = add nuw nsw i32 %254, %141
  %256 = shl nuw nsw i32 %255, 2
  %257 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %256, i32 %250)
  %258 = icmp samesign ult i32 %252, %225
  %259 = bitcast i32 %257 to float
  %260 = fadd contract float %251, %259
  %261 = bitcast float %260 to i32
  %262 = select i1 %258, i32 %261, i32 %250
  %263 = add nuw nsw i32 %226, 8
  %264 = or i32 %206, 32
  %265 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %264, i32 %262)
  %266 = icmp samesign ult i32 %263, %225
  %267 = select i1 %258, float %260, float %251
  %268 = bitcast i32 %265 to float
  %269 = fadd contract float %267, %268
  %270 = select i1 %266, float %269, float %267
  %271 = select i1 %234, float %270, float %223
  br label %272

272:                                              ; preds = %219, %221
  %273 = phi float [ %271, %221 ], [ %214, %219 ]
  %274 = tail call i64 @__ockl_get_local_id(i32 noundef 0) #7
  %275 = and i64 %274, 4294967295
  %276 = icmp eq i64 %275, 0
  br i1 %276, label %277, label %282

277:                                              ; preds = %272
  %278 = sitofp i32 %11 to float
  %279 = fdiv contract float %273, %278
  %280 = fadd contract float %9, %279
  %281 = tail call contract noundef float @__ocml_rsqrt_f32(float noundef %280) #8
  store float %281, ptr addrspace(3) @_ZZN4vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi2ELb1EEEvPT_PKS3_lllllS6_lfiiE10s_variance, align 4, !tbaa !16
  br label %282

282:                                              ; preds = %277, %272
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %283 = tail call i64 @__ockl_get_group_id(i32 noundef 0) #7
  %284 = trunc i64 %283 to i32
  %285 = mul i32 %11, %284
  %286 = zext i32 %285 to i64
  %287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %286
  %288 = tail call i64 @__ockl_get_local_id(i32 noundef 0) #7
  %289 = trunc i64 %288 to i32
  %290 = sdiv i32 %11, 8
  %291 = icmp sgt i32 %290, %289
  br i1 %291, label %292, label %295

292:                                              ; preds = %282
  %293 = load <1 x float>, ptr addrspace(3) @_ZZN4vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi2ELb1EEEvPT_PKS3_lllllS6_lfiiE10s_variance, align 4
  %294 = shufflevector <1 x float> %293, <1 x float> poison, <4 x i32> zeroinitializer
  br label %296

295:                                              ; preds = %296, %282
  ret void

296:                                              ; preds = %292, %296
  %297 = phi i32 [ %289, %292 ], [ %345, %296 ]
  %298 = sext i32 %297 to i64
  %299 = getelementptr inbounds [16 x i8], ptr addrspace(1) %16, i64 %298
  %300 = getelementptr inbounds nuw i8, ptr addrspace(1) %299, i64 8
  %301 = getelementptr inbounds [16 x i8], ptr addrspace(1) %19, i64 %298
  %302 = getelementptr inbounds nuw i8, ptr addrspace(1) %301, i64 8
  %303 = getelementptr inbounds [16 x i8], ptr addrspace(1) %287, i64 %298
  %304 = load <4 x i16>, ptr addrspace(1) %299, align 16
  %305 = load <4 x i16>, ptr addrspace(1) %301, align 16
  %306 = zext <4 x i16> %304 to <4 x i32>
  %307 = shl nuw <4 x i32> %306, splat (i32 16)
  %308 = bitcast <4 x i32> %307 to <4 x float>
  %309 = zext <4 x i16> %305 to <4 x i32>
  %310 = shl nuw <4 x i32> %309, splat (i32 16)
  %311 = bitcast <4 x i32> %310 to <4 x float>
  %312 = fmul contract <4 x float> %294, %308
  %313 = fmul contract <4 x float> %312, %311
  %314 = fcmp contract uno <4 x float> %313, zeroinitializer
  %315 = bitcast <4 x float> %313 to <4 x i32>
  %316 = lshr <4 x i32> %315, splat (i32 16)
  %317 = and <4 x i32> %316, splat (i32 1)
  %318 = add <4 x i32> %315, splat (i32 32767)
  %319 = add <4 x i32> %318, %317
  %320 = lshr <4 x i32> %319, splat (i32 16)
  %321 = trunc nuw <4 x i32> %320 to <4 x i16>
  %322 = select <4 x i1> %314, <4 x i16> splat (i16 32704), <4 x i16> %321
  store <4 x i16> %322, ptr addrspace(1) %303, align 16
  %323 = getelementptr inbounds nuw i8, ptr addrspace(1) %303, i64 8
  %324 = load <4 x i16>, ptr addrspace(1) %300, align 8
  %325 = load <4 x i16>, ptr addrspace(1) %302, align 8
  %326 = zext <4 x i16> %324 to <4 x i32>
  %327 = shl nuw <4 x i32> %326, splat (i32 16)
  %328 = bitcast <4 x i32> %327 to <4 x float>
  %329 = zext <4 x i16> %325 to <4 x i32>
  %330 = shl nuw <4 x i32> %329, splat (i32 16)
  %331 = bitcast <4 x i32> %330 to <4 x float>
  %332 = fmul contract <4 x float> %294, %328
  %333 = fmul contract <4 x float> %332, %331
  %334 = fcmp contract uno <4 x float> %333, zeroinitializer
  %335 = bitcast <4 x float> %333 to <4 x i32>
  %336 = lshr <4 x i32> %335, splat (i32 16)
  %337 = and <4 x i32> %336, splat (i32 1)
  %338 = add <4 x i32> %335, splat (i32 32767)
  %339 = add <4 x i32> %338, %337
  %340 = lshr <4 x i32> %339, splat (i32 16)
  %341 = trunc nuw <4 x i32> %340 to <4 x i16>
  %342 = select <4 x i1> %334, <4 x i16> splat (i16 32704), <4 x i16> %341
  store <4 x i16> %342, ptr addrspace(1) %323, align 8
  %343 = tail call i64 @__ockl_get_local_size(i32 noundef 0) #7
  %344 = trunc i64 %343 to i32
  %345 = add i32 %297, %344
  %346 = icmp slt i32 %345, %290
  br i1 %346, label %296, label %295, !llvm.loop !18
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare hidden i64 @__ockl_get_group_id(i32 noundef) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare hidden i64 @__ockl_get_local_id(i32 noundef) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare hidden i64 @__ockl_get_local_size(i32 noundef) local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #2

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #3

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(read)
declare hidden float @__ocml_rsqrt_f32(float noundef) local_unnamed_addr #4

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.usub.sat.i32(i32, i32) #6

attributes #0 = { convergent mustprogress nofree norecurse nounwind "amdgpu-flat-work-group-size"="1,1024" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx942" "uniform-work-group-size" }
attributes #1 = { convergent mustprogress nofree nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx942" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #3 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nofree nounwind willreturn memory(read) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx942" "uniform-work-group-size" }
attributes #5 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { convergent nounwind willreturn memory(none) "uniform-work-group-size" }
attributes #8 = { convergent nounwind willreturn memory(read) "uniform-work-group-size" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}
!llvm.errno.tbaa = !{!4}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{!"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.mustprogress"}
!10 = !{!11, !12, i64 0}
!11 = !{!"_ZTSN3c108BFloat16E", !12, i64 0}
!12 = !{!"short", !6, i64 0}
!13 = distinct !{!13, !9}
!14 = distinct !{!14, !9}
!15 = distinct !{!15, !9}
!16 = !{!17, !17, i64 0}
!17 = !{!"float", !6, i64 0}
!18 = distinct !{!18, !9}
