; ModuleID = '/root/sess/20261014T183414Z-68d6f6/qwen/b_kept/imp/attention.hip'
source_filename = "/root/sess/20261014T183414Z-68d6f6/qwen/b_kept/imp/attention.hip"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

$_Z39paged_attention_ll4mi_QKV_mfma16_kernelI14__hip_bfloat16S0_LN4vllm18Fp8KVCacheDataTypeE0ES0_Li16ELi128ELi256ELb0ELi8EL8MFMAType0EEvPKT_PKT0_S9_ifPKiSB_SB_iPKfiiiPfSE_PS4_PT2_iSD_SD_ = comdat any

$_Z35paged_attention_ll4mi_reduce_kernelI14__hip_bfloat16S0_Li128ELi128ELi256ELi1EEvPT0_PKfS4_PKT_PKiS9_iS4_ = comdat any

$_ZZ39paged_attention_ll4mi_QKV_mfma16_kernelI14__hip_bfloat16S0_LN4vllm18Fp8KVCacheDataTypeE0ES0_Li16ELi128ELi256ELb0ELi8EL8MFMAType0EEvPKT_PKT0_S9_ifPKiSB_SB_iPKfiiiPfSE_PS4_PT2_iSD_SD_E13shared_logits = comdat any

$_ZZ35paged_attention_ll4mi_reduce_kernelI14__hip_bfloat16S0_Li128ELi128ELi256ELi1EEvPT0_PKfS4_PKT_PKiS9_iS4_E21shared_global_exp_sum = comdat any

$_ZZ35paged_attention_ll4mi_reduce_kernelI14__hip_bfloat16S0_Li128ELi128ELi256ELi1EEvPT0_PKfS4_PKT_PKiS9_iS4_E15shared_exp_sums = comdat any

@_ZZ35paged_attention_ll4mi_reduce_kernelI14__hip_bfloat16S0_Li128ELi128ELi256ELi1EEvPT0_PKfS4_PKT_PKiS9_iS4_E21shared_global_exp_sum = linkonce_odr hidden local_unnamed_addr addrspace(3) global float undef, comdat, align 4
@_ZZ35paged_attention_ll4mi_reduce_kernelI14__hip_bfloat16S0_Li128ELi128ELi256ELi1EEvPT0_PKfS4_PKT_PKiS9_iS4_E15shared_exp_sums = linkonce_odr hidden local_unnamed_addr addrspace(3) global [64 x float] undef, comdat, align 16

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p5(ptr addrspace(5) captures(none)) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p5(ptr addrspace(5) captures(none)) #0

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #1

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #3

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare hidden i64 @__ockl_get_local_id(i32 noundef) local_unnamed_addr #5

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare hidden i64 @__ockl_get_group_id(i32 noundef) local_unnamed_addr #5

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare hidden i64 @__ockl_get_num_groups(i32 noundef) local_unnamed_addr #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #6

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16>, <4 x i16>, <4 x float>, i32 immarg, i32 immarg, i32 immarg) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #1

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #1

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #3 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nofree norecurse nounwind willreturn "amdgpu-flat-work-group-size"="1,256" "amdgpu-waves-per-eu"="5" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx942" "uniform-work-group-size" }
attributes #5 = { convergent mustprogress nofree nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx942" "uniform-work-group-size" }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #7 = { convergent mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #8 = { convergent mustprogress nofree norecurse nounwind willreturn "amdgpu-flat-work-group-size"="1,128" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx942" "uniform-work-group-size" }
attributes #9 = { convergent nounwind willreturn memory(none) "uniform-work-group-size" }
attributes #10 = { nounwind }

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
!8 = !{!6, !6, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"float", !6, i64 0}

@etx_lds_arena = external addrspace(3) global [0 x i8], align 16
declare void @etx_imp_slot_barrier(i32, i32) #12

define internal void @etx_imp_attn_mfma16_g8.body.s0(ptr addrspace(1) noalias noundef readonly captures(none) %0, ptr addrspace(1) noalias noundef readonly captures(none) %1, ptr addrspace(1) noalias noundef readonly captures(none) %2, i32 noundef %3, float noundef %4, ptr addrspace(1) noalias noundef readonly captures(none) %5, ptr addrspace(1) noalias noundef readonly captures(none) %6, ptr addrspace(1) noalias noundef readonly captures(address_is_null) %7, i32 noundef %8, ptr addrspace(1) noalias noundef readnone captures(none) %9, i32 noundef %10, i32 noundef %11, i32 noundef %12, ptr addrspace(1) noalias noundef writeonly captures(none) %13, ptr addrspace(1) noalias noundef writeonly captures(none) %14, ptr addrspace(1) noalias noundef writeonly captures(none) %15, ptr addrspace(1) noalias noundef readnone captures(none) %16, i32 noundef %17, ptr addrspace(1) noundef readnone captures(none) %18, ptr addrspace(1) noundef readnone captures(none) %19, i32 %etx.bx, i32 %etx.by, i32 %etx.bz, i32 %etx.gx, i32 %etx.gy, i32 %etx.gz, i32 %etx.tid, i32 %etx.slot) local_unnamed_addr #11  {
  %etx.lx = urem i32 %etx.tid, 256
  %etx.t1 = udiv i32 %etx.tid, 256
  %etx.ly = urem i32 %etx.t1, 1
  %etx.lz = udiv i32 %etx.t1, 1
  %21 = alloca [4 x float], align 16, addrspace(5)
  %22 = zext i32 %etx.lx to i64
  %23 = trunc i64 %22 to i32
  %24 = lshr i32 %23, 6
  %25 = and i32 %23, 63
  %26 = and i32 %23, 15
  %27 = lshr i32 %25, 4
  %28 = zext i32 %etx.bx to i64
  %29 = trunc i64 %28 to i32
  %30 = icmp eq ptr addrspace(1) %7, null
  br i1 %30, label %31, label %33

31:                                               ; preds = %20
  %32 = and i64 %28, 4294967295
  br label %43

33:                                               ; preds = %20
  %34 = add i64 %28, 1
  %35 = and i64 %34, 4294967295
  %36 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %7, i64 %35
  %37 = load i32, ptr addrspace(1) %36, align 4, !tbaa !4
  %38 = and i64 %28, 4294967295
  %39 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %7, i64 %38
  %40 = load i32, ptr addrspace(1) %39, align 4, !tbaa !4
  %41 = sub nsw i32 %37, %40
  %42 = icmp eq i32 %41, 1
  br i1 %42, label %43, label %1135

43:                                               ; preds = %31, %33
  %44 = phi i64 [ %32, %31 ], [ %38, %33 ]
  %45 = zext i32 %etx.by to i64
  %46 = trunc i64 %45 to i32
  %47 = zext i32 %etx.gy to i64
  %48 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %6, i64 %44
  %49 = load i32, ptr addrspace(1) %48, align 4, !tbaa !4
  %50 = shl i32 %46, 8
  %51 = icmp slt i32 %50, %49
  br i1 %51, label %52, label %1135

52:                                               ; preds = %43
  %53 = zext i32 %etx.bz to i64
  %54 = trunc i64 %53 to i32
  %55 = zext i32 %etx.gz to i64
  %56 = trunc i64 %55 to i32
  %57 = add nsw i32 %49, 15
  %58 = sdiv i32 %57, 16
  %59 = add nsw i32 %58, -1
  %60 = mul i32 %8, %29
  %61 = zext i32 %60 to i64
  %62 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %5, i64 %61
  %63 = and i32 %23, -49
  %64 = add i32 %63, %50
  %65 = icmp slt i32 %64, %49
  %66 = sdiv i32 %64, 16
  %67 = select i1 %65, i32 %66, i32 %59
  %68 = sext i32 %67 to i64
  %69 = getelementptr inbounds [4 x i8], ptr addrspace(1) %62, i64 %68
  %70 = load i32, ptr addrspace(1) %69, align 4, !tbaa !4
  %71 = or disjoint i32 %64, 16
  %72 = icmp slt i32 %71, %49
  %73 = sdiv i32 %71, 16
  %74 = select i1 %72, i32 %73, i32 %59
  %75 = sext i32 %74 to i64
  %76 = getelementptr inbounds [4 x i8], ptr addrspace(1) %62, i64 %75
  %77 = load i32, ptr addrspace(1) %76, align 4, !tbaa !4
  %78 = or disjoint i32 %64, 32
  %79 = icmp slt i32 %78, %49
  %80 = sdiv i32 %78, 16
  %81 = select i1 %79, i32 %80, i32 %59
  %82 = sext i32 %81 to i64
  %83 = getelementptr inbounds [4 x i8], ptr addrspace(1) %62, i64 %82
  %84 = load i32, ptr addrspace(1) %83, align 4, !tbaa !4
  %85 = or disjoint i32 %64, 48
  %86 = icmp slt i32 %85, %49
  %87 = sdiv i32 %85, 16
  %88 = select i1 %86, i32 %87, i32 %59
  %89 = sext i32 %88 to i64
  %90 = getelementptr inbounds [4 x i8], ptr addrspace(1) %62, i64 %89
  %91 = load i32, ptr addrspace(1) %90, align 4, !tbaa !4
  %92 = shl i32 %54, 3
  %93 = shl i32 %56, 3
  %94 = shl nuw nsw i32 %24, 2
  %95 = or disjoint i32 %94, %27
  %96 = add i32 %92, %95
  br i1 %30, label %100, label %97

97:                                               ; preds = %52
  %98 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %7, i64 %44
  %99 = load i32, ptr addrspace(1) %98, align 4, !tbaa !4
  br label %100

100:                                              ; preds = %52, %97
  %101 = phi i32 [ %99, %97 ], [ %29, %52 ]
  %102 = shl nuw nsw i32 %26, 3
  %103 = icmp ult i32 %23, 128
  br i1 %103, label %104, label %123

104:                                              ; preds = %100
  %105 = zext i32 %101 to i64
  %106 = sext i32 %10 to i64
  %107 = mul nsw i64 %105, %106
  %108 = getelementptr inbounds [2 x i8], ptr addrspace(1) %0, i64 %107
  %109 = shl nsw i32 %96, 7
  %110 = sext i32 %109 to i64
  %111 = getelementptr inbounds [2 x i8], ptr addrspace(1) %108, i64 %110
  %112 = zext nneg i32 %102 to i64
  %113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %111, i64 %112
  %114 = load <4 x i16>, ptr addrspace(1) %113, align 8
  %115 = getelementptr inbounds nuw i8, ptr addrspace(1) %113, i64 8
  %116 = load <4 x i16>, ptr addrspace(1) %115, align 8, !tbaa !8
  %117 = lshr i32 %26, 2
  %118 = getelementptr inbounds nuw [2048 x i8], ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @etx_lds_arena, i32 0), i32 %117
  %119 = and i32 %23, 3
  %120 = getelementptr inbounds nuw [512 x i8], ptr addrspace(3) %118, i32 %119
  %121 = getelementptr inbounds nuw [32 x i8], ptr addrspace(3) %120, i32 %95
  store <4 x i16> %114, ptr addrspace(3) %121, align 16, !tbaa !8
  %122 = getelementptr inbounds nuw i8, ptr addrspace(3) %121, i32 8
  store <4 x i16> %116, ptr addrspace(3) %122, align 8, !tbaa !8
  br label %123

123:                                              ; preds = %104, %100
  fence syncscope("workgroup") release
  call void @etx_imp_slot_barrier(i32 %etx.slot, i32 4)
  fence syncscope("workgroup") acquire
  %124 = getelementptr inbounds nuw [512 x i8], ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @etx_lds_arena, i32 0), i32 %27
  %125 = and i32 %23, 7
  %126 = getelementptr inbounds nuw [32 x i8], ptr addrspace(3) %124, i32 %125
  %127 = load <4 x i16>, ptr addrspace(3) %126, align 16, !tbaa !8
  %128 = getelementptr inbounds nuw i8, ptr addrspace(3) %126, i32 8
  %129 = load <4 x i16>, ptr addrspace(3) %128, align 8, !tbaa !8
  %130 = getelementptr inbounds nuw i8, ptr addrspace(3) %126, i32 2048
  %131 = load <4 x i16>, ptr addrspace(3) %130, align 16, !tbaa !8
  %132 = getelementptr inbounds nuw i8, ptr addrspace(3) %126, i32 2056
  %133 = load <4 x i16>, ptr addrspace(3) %132, align 8, !tbaa !8
  %134 = getelementptr inbounds nuw i8, ptr addrspace(3) %126, i32 4096
  %135 = load <4 x i16>, ptr addrspace(3) %134, align 16, !tbaa !8
  %136 = getelementptr inbounds nuw i8, ptr addrspace(3) %126, i32 4104
  %137 = load <4 x i16>, ptr addrspace(3) %136, align 8, !tbaa !8
  %138 = getelementptr inbounds nuw i8, ptr addrspace(3) %126, i32 6144
  %139 = load <4 x i16>, ptr addrspace(3) %138, align 16, !tbaa !8
  %140 = getelementptr inbounds nuw i8, ptr addrspace(3) %126, i32 6152
  %141 = load <4 x i16>, ptr addrspace(3) %140, align 8, !tbaa !8
  %142 = mul i32 %12, %54
  %143 = zext i32 %142 to i64
  %144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1, i64 %143
  %145 = sext i32 %11 to i64
  %146 = srem i32 %23, 16
  %147 = shl nsw i32 %146, 3
  %148 = sext i32 %147 to i64
  %149 = getelementptr [2 x i8], ptr addrspace(1) %144, i64 %148
  %150 = shl i32 %23, 3
  %151 = and i32 %150, 384
  %152 = zext nneg i32 %151 to i64
  %153 = sext i32 %70 to i64
  %154 = mul nsw i64 %153, %145
  %155 = getelementptr [2 x i8], ptr addrspace(1) %149, i64 %154
  %156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %155, i64 %152
  %157 = load <4 x i16>, ptr addrspace(1) %156, align 8
  %158 = getelementptr inbounds nuw i8, ptr addrspace(1) %156, i64 8
  %159 = load <4 x i16>, ptr addrspace(1) %158, align 8, !tbaa !8
  %160 = or disjoint i32 %151, 512
  %161 = zext nneg i32 %160 to i64
  %162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %155, i64 %161
  %163 = load <4 x i16>, ptr addrspace(1) %162, align 8
  %164 = getelementptr inbounds nuw i8, ptr addrspace(1) %162, i64 8
  %165 = load <4 x i16>, ptr addrspace(1) %164, align 8, !tbaa !8
  %166 = or disjoint i32 %151, 1024
  %167 = zext nneg i32 %166 to i64
  %168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %155, i64 %167
  %169 = load <4 x i16>, ptr addrspace(1) %168, align 8
  %170 = getelementptr inbounds nuw i8, ptr addrspace(1) %168, i64 8
  %171 = load <4 x i16>, ptr addrspace(1) %170, align 8, !tbaa !8
  %172 = or disjoint i32 %151, 1536
  %173 = zext nneg i32 %172 to i64
  %174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %155, i64 %173
  %175 = load <4 x i16>, ptr addrspace(1) %174, align 8
  %176 = getelementptr inbounds nuw i8, ptr addrspace(1) %174, i64 8
  %177 = load <4 x i16>, ptr addrspace(1) %176, align 8, !tbaa !8
  %178 = sext i32 %77 to i64
  %179 = mul nsw i64 %178, %145
  %180 = getelementptr [2 x i8], ptr addrspace(1) %149, i64 %179
  %181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %180, i64 %152
  %182 = load <4 x i16>, ptr addrspace(1) %181, align 8
  %183 = getelementptr inbounds nuw i8, ptr addrspace(1) %181, i64 8
  %184 = load <4 x i16>, ptr addrspace(1) %183, align 8, !tbaa !8
  %185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %180, i64 %161
  %186 = load <4 x i16>, ptr addrspace(1) %185, align 8
  %187 = getelementptr inbounds nuw i8, ptr addrspace(1) %185, i64 8
  %188 = load <4 x i16>, ptr addrspace(1) %187, align 8, !tbaa !8
  %189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %180, i64 %167
  %190 = load <4 x i16>, ptr addrspace(1) %189, align 8
  %191 = getelementptr inbounds nuw i8, ptr addrspace(1) %189, i64 8
  %192 = load <4 x i16>, ptr addrspace(1) %191, align 8, !tbaa !8
  %193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %180, i64 %173
  %194 = load <4 x i16>, ptr addrspace(1) %193, align 8
  %195 = getelementptr inbounds nuw i8, ptr addrspace(1) %193, i64 8
  %196 = load <4 x i16>, ptr addrspace(1) %195, align 8, !tbaa !8
  %197 = sext i32 %84 to i64
  %198 = mul nsw i64 %197, %145
  %199 = getelementptr [2 x i8], ptr addrspace(1) %149, i64 %198
  %200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %199, i64 %152
  %201 = load <4 x i16>, ptr addrspace(1) %200, align 8
  %202 = getelementptr inbounds nuw i8, ptr addrspace(1) %200, i64 8
  %203 = load <4 x i16>, ptr addrspace(1) %202, align 8, !tbaa !8
  %204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %199, i64 %161
  %205 = load <4 x i16>, ptr addrspace(1) %204, align 8
  %206 = getelementptr inbounds nuw i8, ptr addrspace(1) %204, i64 8
  %207 = load <4 x i16>, ptr addrspace(1) %206, align 8, !tbaa !8
  %208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %199, i64 %167
  %209 = load <4 x i16>, ptr addrspace(1) %208, align 8
  %210 = getelementptr inbounds nuw i8, ptr addrspace(1) %208, i64 8
  %211 = load <4 x i16>, ptr addrspace(1) %210, align 8, !tbaa !8
  %212 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %199, i64 %173
  %213 = load <4 x i16>, ptr addrspace(1) %212, align 8
  %214 = getelementptr inbounds nuw i8, ptr addrspace(1) %212, i64 8
  %215 = load <4 x i16>, ptr addrspace(1) %214, align 8, !tbaa !8
  %216 = sext i32 %91 to i64
  %217 = mul nsw i64 %216, %145
  %218 = getelementptr [2 x i8], ptr addrspace(1) %149, i64 %217
  %219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %218, i64 %152
  %220 = load <4 x i16>, ptr addrspace(1) %219, align 8
  %221 = getelementptr inbounds nuw i8, ptr addrspace(1) %219, i64 8
  %222 = load <4 x i16>, ptr addrspace(1) %221, align 8, !tbaa !8
  %223 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %218, i64 %161
  %224 = load <4 x i16>, ptr addrspace(1) %223, align 8
  %225 = getelementptr inbounds nuw i8, ptr addrspace(1) %223, i64 8
  %226 = load <4 x i16>, ptr addrspace(1) %225, align 8, !tbaa !8
  %227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %218, i64 %167
  %228 = load <4 x i16>, ptr addrspace(1) %227, align 8
  %229 = getelementptr inbounds nuw i8, ptr addrspace(1) %227, i64 8
  %230 = load <4 x i16>, ptr addrspace(1) %229, align 8, !tbaa !8
  %231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %218, i64 %173
  %232 = load <4 x i16>, ptr addrspace(1) %231, align 8
  %233 = getelementptr inbounds nuw i8, ptr addrspace(1) %231, i64 8
  %234 = load <4 x i16>, ptr addrspace(1) %233, align 8, !tbaa !8
  %235 = and i32 %23, 48
  %236 = or disjoint i32 %235, %50
  %237 = icmp slt i32 %236, %49
  %238 = ashr exact i32 %236, 4
  %239 = select i1 %237, i32 %238, i32 %59
  %240 = sext i32 %239 to i64
  %241 = getelementptr inbounds [4 x i8], ptr addrspace(1) %62, i64 %240
  %242 = load i32, ptr addrspace(1) %241, align 4, !tbaa !4
  %243 = or disjoint i32 %236, 64
  %244 = icmp slt i32 %243, %49
  %245 = ashr exact i32 %243, 4
  %246 = select i1 %244, i32 %245, i32 %59
  %247 = sext i32 %246 to i64
  %248 = getelementptr inbounds [4 x i8], ptr addrspace(1) %62, i64 %247
  %249 = load i32, ptr addrspace(1) %248, align 4, !tbaa !4
  %250 = or disjoint i32 %236, 128
  %251 = icmp slt i32 %250, %49
  %252 = ashr exact i32 %250, 4
  %253 = select i1 %251, i32 %252, i32 %59
  %254 = sext i32 %253 to i64
  %255 = getelementptr inbounds [4 x i8], ptr addrspace(1) %62, i64 %254
  %256 = load i32, ptr addrspace(1) %255, align 4, !tbaa !4
  %257 = or disjoint i32 %236, 192
  %258 = icmp slt i32 %257, %49
  %259 = ashr exact i32 %257, 4
  %260 = select i1 %258, i32 %259, i32 %59
  %261 = sext i32 %260 to i64
  %262 = getelementptr inbounds [4 x i8], ptr addrspace(1) %62, i64 %261
  %263 = load i32, ptr addrspace(1) %262, align 4, !tbaa !4
  %264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2, i64 %143
  %265 = shl i32 %24, 8
  %266 = shl nuw nsw i32 %26, 4
  %267 = or disjoint i32 %265, %266
  %268 = zext nneg i32 %267 to i64
  %269 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %264, i64 %268
  %270 = sext i32 %242 to i64
  %271 = mul nsw i64 %270, %145
  %272 = getelementptr inbounds [2 x i8], ptr addrspace(1) %269, i64 %271
  %273 = load <4 x i16>, ptr addrspace(1) %272, align 8
  %274 = getelementptr inbounds nuw i8, ptr addrspace(1) %272, i64 8
  %275 = load <4 x i16>, ptr addrspace(1) %274, align 8, !tbaa !8
  %276 = getelementptr inbounds nuw i8, ptr addrspace(1) %272, i64 16
  %277 = load <4 x i16>, ptr addrspace(1) %276, align 8
  %278 = getelementptr inbounds nuw i8, ptr addrspace(1) %272, i64 24
  %279 = load <4 x i16>, ptr addrspace(1) %278, align 8, !tbaa !8
  %280 = sext i32 %249 to i64
  %281 = mul nsw i64 %280, %145
  %282 = getelementptr inbounds [2 x i8], ptr addrspace(1) %269, i64 %281
  %283 = load <4 x i16>, ptr addrspace(1) %282, align 8
  %284 = getelementptr inbounds nuw i8, ptr addrspace(1) %282, i64 8
  %285 = load <4 x i16>, ptr addrspace(1) %284, align 8, !tbaa !8
  %286 = getelementptr inbounds nuw i8, ptr addrspace(1) %282, i64 16
  %287 = load <4 x i16>, ptr addrspace(1) %286, align 8
  %288 = getelementptr inbounds nuw i8, ptr addrspace(1) %282, i64 24
  %289 = load <4 x i16>, ptr addrspace(1) %288, align 8, !tbaa !8
  %290 = sext i32 %256 to i64
  %291 = mul nsw i64 %290, %145
  %292 = getelementptr inbounds [2 x i8], ptr addrspace(1) %269, i64 %291
  %293 = load <4 x i16>, ptr addrspace(1) %292, align 8
  %294 = getelementptr inbounds nuw i8, ptr addrspace(1) %292, i64 8
  %295 = load <4 x i16>, ptr addrspace(1) %294, align 8, !tbaa !8
  %296 = getelementptr inbounds nuw i8, ptr addrspace(1) %292, i64 16
  %297 = load <4 x i16>, ptr addrspace(1) %296, align 8
  %298 = getelementptr inbounds nuw i8, ptr addrspace(1) %292, i64 24
  %299 = load <4 x i16>, ptr addrspace(1) %298, align 8, !tbaa !8
  %300 = sext i32 %263 to i64
  %301 = mul nsw i64 %300, %145
  %302 = getelementptr inbounds [2 x i8], ptr addrspace(1) %269, i64 %301
  %303 = load <4 x i16>, ptr addrspace(1) %302, align 8
  %304 = getelementptr inbounds nuw i8, ptr addrspace(1) %302, i64 8
  %305 = load <4 x i16>, ptr addrspace(1) %304, align 8, !tbaa !8
  %306 = getelementptr inbounds nuw i8, ptr addrspace(1) %302, i64 16
  %307 = load <4 x i16>, ptr addrspace(1) %306, align 8
  %308 = getelementptr inbounds nuw i8, ptr addrspace(1) %302, i64 24
  %309 = load <4 x i16>, ptr addrspace(1) %308, align 8, !tbaa !8
  %310 = add i32 %267, 1024
  %311 = zext nneg i32 %310 to i64
  %312 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %264, i64 %311
  %313 = getelementptr inbounds [2 x i8], ptr addrspace(1) %312, i64 %271
  %314 = load <4 x i16>, ptr addrspace(1) %313, align 8
  %315 = getelementptr inbounds nuw i8, ptr addrspace(1) %313, i64 8
  %316 = load <4 x i16>, ptr addrspace(1) %315, align 8, !tbaa !8
  %317 = getelementptr inbounds nuw i8, ptr addrspace(1) %313, i64 16
  %318 = load <4 x i16>, ptr addrspace(1) %317, align 8
  %319 = getelementptr inbounds nuw i8, ptr addrspace(1) %313, i64 24
  %320 = load <4 x i16>, ptr addrspace(1) %319, align 8, !tbaa !8
  %321 = getelementptr inbounds [2 x i8], ptr addrspace(1) %312, i64 %281
  %322 = load <4 x i16>, ptr addrspace(1) %321, align 8
  %323 = getelementptr inbounds nuw i8, ptr addrspace(1) %321, i64 8
  %324 = load <4 x i16>, ptr addrspace(1) %323, align 8, !tbaa !8
  %325 = getelementptr inbounds nuw i8, ptr addrspace(1) %321, i64 16
  %326 = load <4 x i16>, ptr addrspace(1) %325, align 8
  %327 = getelementptr inbounds nuw i8, ptr addrspace(1) %321, i64 24
  %328 = load <4 x i16>, ptr addrspace(1) %327, align 8, !tbaa !8
  %329 = getelementptr inbounds [2 x i8], ptr addrspace(1) %312, i64 %291
  %330 = load <4 x i16>, ptr addrspace(1) %329, align 8
  %331 = getelementptr inbounds nuw i8, ptr addrspace(1) %329, i64 8
  %332 = load <4 x i16>, ptr addrspace(1) %331, align 8, !tbaa !8
  %333 = getelementptr inbounds nuw i8, ptr addrspace(1) %329, i64 16
  %334 = load <4 x i16>, ptr addrspace(1) %333, align 8
  %335 = getelementptr inbounds nuw i8, ptr addrspace(1) %329, i64 24
  %336 = load <4 x i16>, ptr addrspace(1) %335, align 8, !tbaa !8
  %337 = getelementptr inbounds [2 x i8], ptr addrspace(1) %312, i64 %301
  %338 = load <4 x i16>, ptr addrspace(1) %337, align 8
  %339 = getelementptr inbounds nuw i8, ptr addrspace(1) %337, i64 8
  %340 = load <4 x i16>, ptr addrspace(1) %339, align 8, !tbaa !8
  %341 = getelementptr inbounds nuw i8, ptr addrspace(1) %337, i64 16
  %342 = load <4 x i16>, ptr addrspace(1) %341, align 8
  %343 = getelementptr inbounds nuw i8, ptr addrspace(1) %337, i64 24
  %344 = load <4 x i16>, ptr addrspace(1) %343, align 8, !tbaa !8
  %345 = insertelement <4 x float> poison, float %4, i64 0
  %346 = shufflevector <4 x float> %345, <4 x float> poison, <4 x i32> zeroinitializer
  %347 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %157, <4 x i16> %127, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %348 = shufflevector <4 x float> %347, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %349 = shufflevector <16 x float> %348, <16 x float> <float poison, float poison, float poison, float poison, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %350 = bitcast <16 x float> %349 to <4 x i128>
  %351 = extractelement <4 x i128> %350, i64 0
  %352 = bitcast i128 %351 to <4 x float>
  %353 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %159, <4 x i16> %129, <4 x float> %352, i32 0, i32 0, i32 0)
  %354 = shufflevector <4 x float> %353, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %355 = shufflevector <16 x float> %354, <16 x float> %349, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %356 = bitcast <16 x float> %355 to <4 x i128>
  %357 = extractelement <4 x i128> %356, i64 0
  %358 = bitcast i128 %357 to <4 x float>
  %359 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %163, <4 x i16> %131, <4 x float> %358, i32 0, i32 0, i32 0)
  %360 = shufflevector <4 x float> %359, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %361 = shufflevector <16 x float> %360, <16 x float> %355, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %362 = bitcast <16 x float> %361 to <4 x i128>
  %363 = extractelement <4 x i128> %362, i64 0
  %364 = bitcast i128 %363 to <4 x float>
  %365 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %165, <4 x i16> %133, <4 x float> %364, i32 0, i32 0, i32 0)
  %366 = shufflevector <4 x float> %365, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %367 = shufflevector <16 x float> %366, <16 x float> %361, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %368 = bitcast <16 x float> %367 to <4 x i128>
  %369 = extractelement <4 x i128> %368, i64 0
  %370 = bitcast i128 %369 to <4 x float>
  %371 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %169, <4 x i16> %135, <4 x float> %370, i32 0, i32 0, i32 0)
  %372 = shufflevector <4 x float> %371, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %373 = shufflevector <16 x float> %372, <16 x float> %367, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %374 = bitcast <16 x float> %373 to <4 x i128>
  %375 = extractelement <4 x i128> %374, i64 0
  %376 = bitcast i128 %375 to <4 x float>
  %377 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %171, <4 x i16> %137, <4 x float> %376, i32 0, i32 0, i32 0)
  %378 = shufflevector <4 x float> %377, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %379 = shufflevector <16 x float> %378, <16 x float> %373, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %380 = bitcast <16 x float> %379 to <4 x i128>
  %381 = extractelement <4 x i128> %380, i64 0
  %382 = bitcast i128 %381 to <4 x float>
  %383 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %175, <4 x i16> %139, <4 x float> %382, i32 0, i32 0, i32 0)
  %384 = shufflevector <4 x float> %383, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %385 = bitcast <16 x float> %384 to <4 x i128>
  %386 = extractelement <4 x i128> %385, i64 0
  %387 = bitcast i128 %386 to <4 x float>
  %388 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %177, <4 x i16> %141, <4 x float> %387, i32 0, i32 0, i32 0)
  %389 = shufflevector <4 x float> %388, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %390 = bitcast <16 x float> %389 to <4 x i128>
  %391 = extractelement <4 x i128> %390, i64 0
  %392 = bitcast i128 %391 to <4 x float>
  %393 = fmul contract <4 x float> %346, %392
  %394 = shufflevector <4 x float> %393, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %395 = shufflevector <16 x float> %394, <16 x float> <float poison, float poison, float poison, float poison, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %396 = bitcast <16 x float> %395 to <4 x i128>
  %397 = extractelement <4 x i128> %396, i64 1
  %398 = bitcast i128 %397 to <4 x float>
  %399 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %182, <4 x i16> %127, <4 x float> %398, i32 0, i32 0, i32 0)
  %400 = shufflevector <4 x float> %399, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %401 = shufflevector <16 x float> %395, <16 x float> %400, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %402 = bitcast <16 x float> %401 to <4 x i128>
  %403 = extractelement <4 x i128> %402, i64 1
  %404 = bitcast i128 %403 to <4 x float>
  %405 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %184, <4 x i16> %129, <4 x float> %404, i32 0, i32 0, i32 0)
  %406 = shufflevector <4 x float> %405, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %407 = shufflevector <16 x float> %401, <16 x float> %406, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %408 = bitcast <16 x float> %407 to <4 x i128>
  %409 = extractelement <4 x i128> %408, i64 1
  %410 = bitcast i128 %409 to <4 x float>
  %411 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %186, <4 x i16> %131, <4 x float> %410, i32 0, i32 0, i32 0)
  %412 = shufflevector <4 x float> %411, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %413 = shufflevector <16 x float> %407, <16 x float> %412, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %414 = bitcast <16 x float> %413 to <4 x i128>
  %415 = extractelement <4 x i128> %414, i64 1
  %416 = bitcast i128 %415 to <4 x float>
  %417 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %188, <4 x i16> %133, <4 x float> %416, i32 0, i32 0, i32 0)
  %418 = shufflevector <4 x float> %417, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %419 = shufflevector <16 x float> %413, <16 x float> %418, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %420 = bitcast <16 x float> %419 to <4 x i128>
  %421 = extractelement <4 x i128> %420, i64 1
  %422 = bitcast i128 %421 to <4 x float>
  %423 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %190, <4 x i16> %135, <4 x float> %422, i32 0, i32 0, i32 0)
  %424 = shufflevector <4 x float> %423, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %425 = shufflevector <16 x float> %419, <16 x float> %424, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %426 = bitcast <16 x float> %425 to <4 x i128>
  %427 = extractelement <4 x i128> %426, i64 1
  %428 = bitcast i128 %427 to <4 x float>
  %429 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %192, <4 x i16> %137, <4 x float> %428, i32 0, i32 0, i32 0)
  %430 = shufflevector <4 x float> %429, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %431 = shufflevector <16 x float> %425, <16 x float> %430, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %432 = bitcast <16 x float> %431 to <4 x i128>
  %433 = extractelement <4 x i128> %432, i64 1
  %434 = bitcast i128 %433 to <4 x float>
  %435 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %194, <4 x i16> %139, <4 x float> %434, i32 0, i32 0, i32 0)
  %436 = shufflevector <4 x float> %435, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %437 = bitcast <16 x float> %436 to <4 x i128>
  %438 = extractelement <4 x i128> %437, i64 1
  %439 = bitcast i128 %438 to <4 x float>
  %440 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %196, <4 x i16> %141, <4 x float> %439, i32 0, i32 0, i32 0)
  %441 = shufflevector <4 x float> %440, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %442 = bitcast <16 x float> %441 to <4 x i128>
  %443 = extractelement <4 x i128> %442, i64 1
  %444 = bitcast i128 %443 to <4 x float>
  %445 = fmul contract <4 x float> %346, %444
  %446 = shufflevector <4 x float> %445, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %447 = shufflevector <16 x float> %395, <16 x float> <float poison, float poison, float poison, float poison, float poison, float poison, float poison, float poison, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float poison, float poison, float poison, float poison>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 26, i32 27, i32 12, i32 13, i32 14, i32 15>
  %448 = shufflevector <16 x float> %447, <16 x float> %446, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %449 = bitcast <16 x float> %448 to <4 x i128>
  %450 = extractelement <4 x i128> %449, i64 2
  %451 = bitcast i128 %450 to <4 x float>
  %452 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %201, <4 x i16> %127, <4 x float> %451, i32 0, i32 0, i32 0)
  %453 = shufflevector <4 x float> %452, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %454 = shufflevector <16 x float> %448, <16 x float> %453, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 12, i32 13, i32 14, i32 15>
  %455 = bitcast <16 x float> %454 to <4 x i128>
  %456 = extractelement <4 x i128> %455, i64 2
  %457 = bitcast i128 %456 to <4 x float>
  %458 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %203, <4 x i16> %129, <4 x float> %457, i32 0, i32 0, i32 0)
  %459 = shufflevector <4 x float> %458, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %460 = shufflevector <16 x float> %454, <16 x float> %459, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 12, i32 13, i32 14, i32 15>
  %461 = bitcast <16 x float> %460 to <4 x i128>
  %462 = extractelement <4 x i128> %461, i64 2
  %463 = bitcast i128 %462 to <4 x float>
  %464 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %205, <4 x i16> %131, <4 x float> %463, i32 0, i32 0, i32 0)
  %465 = shufflevector <4 x float> %464, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %466 = shufflevector <16 x float> %460, <16 x float> %465, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 12, i32 13, i32 14, i32 15>
  %467 = bitcast <16 x float> %466 to <4 x i128>
  %468 = extractelement <4 x i128> %467, i64 2
  %469 = bitcast i128 %468 to <4 x float>
  %470 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %207, <4 x i16> %133, <4 x float> %469, i32 0, i32 0, i32 0)
  %471 = shufflevector <4 x float> %470, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %472 = shufflevector <16 x float> %466, <16 x float> %471, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 12, i32 13, i32 14, i32 15>
  %473 = bitcast <16 x float> %472 to <4 x i128>
  %474 = extractelement <4 x i128> %473, i64 2
  %475 = bitcast i128 %474 to <4 x float>
  %476 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %209, <4 x i16> %135, <4 x float> %475, i32 0, i32 0, i32 0)
  %477 = shufflevector <4 x float> %476, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %478 = shufflevector <16 x float> %472, <16 x float> %477, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 12, i32 13, i32 14, i32 15>
  %479 = bitcast <16 x float> %478 to <4 x i128>
  %480 = extractelement <4 x i128> %479, i64 2
  %481 = bitcast i128 %480 to <4 x float>
  %482 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %211, <4 x i16> %137, <4 x float> %481, i32 0, i32 0, i32 0)
  %483 = shufflevector <4 x float> %482, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %484 = shufflevector <16 x float> %478, <16 x float> %483, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 12, i32 13, i32 14, i32 15>
  %485 = bitcast <16 x float> %484 to <4 x i128>
  %486 = extractelement <4 x i128> %485, i64 2
  %487 = bitcast i128 %486 to <4 x float>
  %488 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %213, <4 x i16> %139, <4 x float> %487, i32 0, i32 0, i32 0)
  %489 = shufflevector <4 x float> %488, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %490 = bitcast <16 x float> %489 to <4 x i128>
  %491 = extractelement <4 x i128> %490, i64 2
  %492 = bitcast i128 %491 to <4 x float>
  %493 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %215, <4 x i16> %141, <4 x float> %492, i32 0, i32 0, i32 0)
  %494 = shufflevector <4 x float> %493, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %495 = bitcast <16 x float> %494 to <4 x i128>
  %496 = extractelement <4 x i128> %495, i64 2
  %497 = bitcast i128 %496 to <4 x float>
  %498 = fmul contract <4 x float> %346, %497
  %499 = shufflevector <4 x float> %498, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %500 = shufflevector <16 x float> %448, <16 x float> <float poison, float poison, float poison, float poison, float poison, float poison, float poison, float poison, float poison, float poison, float poison, float poison, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 28, i32 29, i32 30, i32 31>
  %501 = shufflevector <16 x float> %500, <16 x float> %499, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 12, i32 13, i32 14, i32 15>
  %502 = bitcast <16 x float> %501 to <4 x i128>
  %503 = extractelement <4 x i128> %502, i64 3
  %504 = bitcast i128 %503 to <4 x float>
  %505 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %220, <4 x i16> %127, <4 x float> %504, i32 0, i32 0, i32 0)
  %506 = shufflevector <4 x float> %505, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %507 = shufflevector <16 x float> %501, <16 x float> %506, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>
  %508 = bitcast <16 x float> %507 to <4 x i128>
  %509 = extractelement <4 x i128> %508, i64 3
  %510 = bitcast i128 %509 to <4 x float>
  %511 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %222, <4 x i16> %129, <4 x float> %510, i32 0, i32 0, i32 0)
  %512 = shufflevector <4 x float> %511, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %513 = shufflevector <16 x float> %507, <16 x float> %512, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>
  %514 = bitcast <16 x float> %513 to <4 x i128>
  %515 = extractelement <4 x i128> %514, i64 3
  %516 = bitcast i128 %515 to <4 x float>
  %517 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %224, <4 x i16> %131, <4 x float> %516, i32 0, i32 0, i32 0)
  %518 = shufflevector <4 x float> %517, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %519 = shufflevector <16 x float> %513, <16 x float> %518, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>
  %520 = bitcast <16 x float> %519 to <4 x i128>
  %521 = extractelement <4 x i128> %520, i64 3
  %522 = bitcast i128 %521 to <4 x float>
  %523 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %226, <4 x i16> %133, <4 x float> %522, i32 0, i32 0, i32 0)
  %524 = shufflevector <4 x float> %523, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %525 = shufflevector <16 x float> %519, <16 x float> %524, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>
  %526 = bitcast <16 x float> %525 to <4 x i128>
  %527 = extractelement <4 x i128> %526, i64 3
  %528 = bitcast i128 %527 to <4 x float>
  %529 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %228, <4 x i16> %135, <4 x float> %528, i32 0, i32 0, i32 0)
  %530 = shufflevector <4 x float> %529, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %531 = shufflevector <16 x float> %525, <16 x float> %530, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>
  %532 = bitcast <16 x float> %531 to <4 x i128>
  %533 = extractelement <4 x i128> %532, i64 3
  %534 = bitcast i128 %533 to <4 x float>
  %535 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %230, <4 x i16> %137, <4 x float> %534, i32 0, i32 0, i32 0)
  %536 = shufflevector <4 x float> %535, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %537 = shufflevector <16 x float> %531, <16 x float> %536, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>
  %538 = bitcast <16 x float> %537 to <4 x i128>
  %539 = extractelement <4 x i128> %538, i64 3
  %540 = bitcast i128 %539 to <4 x float>
  %541 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %232, <4 x i16> %139, <4 x float> %540, i32 0, i32 0, i32 0)
  %542 = shufflevector <4 x float> %541, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3>
  %543 = bitcast <16 x float> %542 to <4 x i128>
  %544 = extractelement <4 x i128> %543, i64 3
  %545 = bitcast i128 %544 to <4 x float>
  %546 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %234, <4 x i16> %141, <4 x float> %545, i32 0, i32 0, i32 0)
  %547 = shufflevector <4 x float> %546, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3>
  %548 = bitcast <16 x float> %547 to <4 x i128>
  %549 = extractelement <4 x i128> %548, i64 3
  %550 = bitcast i128 %549 to <4 x float>
  %551 = shufflevector <4 x float> %345, <4 x float> poison, <16 x i32> zeroinitializer
  %552 = shufflevector <4 x float> %392, <4 x float> %444, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %553 = shufflevector <4 x float> %497, <4 x float> %550, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %554 = shufflevector <8 x float> %552, <8 x float> %553, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %555 = fmul contract <16 x float> %551, %554
  %556 = and i32 %23, -64
  %557 = add i32 %50, %556
  %558 = shl nuw nsw i32 %27, 2
  %559 = or disjoint i32 %557, %558
  %560 = bitcast <16 x float> %555 to <4 x i128>
  %561 = extractelement <4 x i128> %560, i64 0
  %562 = bitcast i128 %561 to <4 x float>
  %563 = shufflevector <4 x float> %562, <4 x float> poison, <16 x i32> <i32 poison, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %564 = icmp slt i32 %559, %49
  %565 = extractelement <4 x float> %562, i64 0
  %566 = tail call nsz contract float @llvm.maxnum.f32(float %565, float 0xC7EFFFFFE0000000)
  %567 = select i1 %564, float %566, float 0xC7EFFFFFE0000000
  %568 = or disjoint i32 %559, 1
  %569 = icmp slt i32 %568, %49
  %570 = extractelement <4 x float> %562, i64 1
  %571 = select i1 %569, float %570, float 0xC7EFFFFFE0000000
  %572 = tail call nsz contract noundef float @llvm.maxnum.f32(float %567, float %571)
  %573 = or disjoint i32 %559, 2
  %574 = icmp slt i32 %573, %49
  %575 = extractelement <4 x float> %562, i64 2
  %576 = select i1 %574, float %575, float 0xC7EFFFFFE0000000
  %577 = tail call nsz contract noundef float @llvm.maxnum.f32(float %572, float %576)
  %578 = or disjoint i32 %559, 3
  %579 = icmp slt i32 %578, %49
  %580 = extractelement <4 x float> %562, i64 3
  %581 = select i1 %579, float %580, float 0xC7EFFFFFE0000000
  %582 = tail call nsz contract noundef float @llvm.maxnum.f32(float %577, float %581)
  %583 = or disjoint i32 %559, 16
  %584 = extractelement <4 x i128> %560, i64 1
  %585 = bitcast i128 %584 to <4 x float>
  %586 = icmp slt i32 %583, %49
  %587 = extractelement <4 x float> %585, i64 0
  %588 = select i1 %586, float %587, float 0xC7EFFFFFE0000000
  %589 = tail call nsz contract noundef float @llvm.maxnum.f32(float %582, float %588)
  %590 = or disjoint i32 %559, 17
  %591 = icmp slt i32 %590, %49
  %592 = extractelement <4 x float> %585, i64 1
  %593 = select i1 %591, float %592, float 0xC7EFFFFFE0000000
  %594 = tail call nsz contract noundef float @llvm.maxnum.f32(float %589, float %593)
  %595 = or disjoint i32 %559, 18
  %596 = icmp slt i32 %595, %49
  %597 = extractelement <4 x float> %585, i64 2
  %598 = select i1 %596, float %597, float 0xC7EFFFFFE0000000
  %599 = tail call nsz contract noundef float @llvm.maxnum.f32(float %594, float %598)
  %600 = or disjoint i32 %559, 19
  %601 = icmp slt i32 %600, %49
  %602 = extractelement <4 x float> %585, i64 3
  %603 = select i1 %601, float %602, float 0xC7EFFFFFE0000000
  %604 = tail call nsz contract noundef float @llvm.maxnum.f32(float %599, float %603)
  %605 = or disjoint i32 %559, 32
  %606 = extractelement <4 x i128> %560, i64 2
  %607 = bitcast i128 %606 to <4 x float>
  %608 = icmp slt i32 %605, %49
  %609 = extractelement <4 x float> %607, i64 0
  %610 = select i1 %608, float %609, float 0xC7EFFFFFE0000000
  %611 = tail call nsz contract noundef float @llvm.maxnum.f32(float %604, float %610)
  %612 = or disjoint i32 %559, 33
  %613 = icmp slt i32 %612, %49
  %614 = extractelement <4 x float> %607, i64 1
  %615 = select i1 %613, float %614, float 0xC7EFFFFFE0000000
  %616 = tail call nsz contract noundef float @llvm.maxnum.f32(float %611, float %615)
  %617 = or disjoint i32 %559, 34
  %618 = icmp slt i32 %617, %49
  %619 = extractelement <4 x float> %607, i64 2
  %620 = select i1 %618, float %619, float 0xC7EFFFFFE0000000
  %621 = tail call nsz contract noundef float @llvm.maxnum.f32(float %616, float %620)
  %622 = or disjoint i32 %559, 35
  %623 = icmp slt i32 %622, %49
  %624 = extractelement <4 x float> %607, i64 3
  %625 = select i1 %623, float %624, float 0xC7EFFFFFE0000000
  %626 = tail call nsz contract noundef float @llvm.maxnum.f32(float %621, float %625)
  %627 = or disjoint i32 %559, 48
  %628 = extractelement <4 x i128> %560, i64 3
  %629 = bitcast i128 %628 to <4 x float>
  %630 = icmp slt i32 %627, %49
  %631 = extractelement <4 x float> %629, i64 0
  %632 = select i1 %630, float %631, float 0xC7EFFFFFE0000000
  %633 = tail call nsz contract noundef float @llvm.maxnum.f32(float %626, float %632)
  %634 = or disjoint i32 %559, 49
  %635 = icmp slt i32 %634, %49
  %636 = extractelement <4 x float> %629, i64 1
  %637 = select i1 %635, float %636, float 0xC7EFFFFFE0000000
  %638 = tail call nsz contract noundef float @llvm.maxnum.f32(float %633, float %637)
  %639 = or disjoint i32 %559, 50
  %640 = icmp slt i32 %639, %49
  %641 = extractelement <4 x float> %629, i64 2
  %642 = select i1 %640, float %641, float 0xC7EFFFFFE0000000
  %643 = tail call nsz contract noundef float @llvm.maxnum.f32(float %638, float %642)
  %644 = or disjoint i32 %559, 51
  %645 = icmp slt i32 %644, %49
  %646 = extractelement <4 x float> %629, i64 3
  %647 = select i1 %645, float %646, float 0xC7EFFFFFE0000000
  %648 = tail call nsz contract noundef float @llvm.maxnum.f32(float %643, float %647)
  %649 = tail call range(i32 0, 33) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %650 = tail call noundef range(i32 0, 65) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %649)
  %651 = add nuw nsw i32 %650, 64
  %652 = and i32 %651, 192
  %653 = bitcast float %648 to i32
  %654 = xor i32 %650, 32
  %655 = icmp samesign ult i32 %654, %652
  %656 = select i1 %655, i32 %654, i32 %650
  %657 = shl nuw nsw i32 %656, 2
  %658 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %657, i32 %653)
  %659 = bitcast i32 %658 to float
  %660 = tail call nsz contract noundef float @llvm.maxnum.f32(float %648, float %659)
  %661 = bitcast float %660 to i32
  %662 = xor i32 %650, 16
  %663 = icmp samesign ult i32 %662, %652
  %664 = select i1 %663, i32 %662, i32 %650
  %665 = shl nuw nsw i32 %664, 2
  %666 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %665, i32 %661)
  %667 = bitcast i32 %666 to float
  %668 = tail call nsz contract noundef float @llvm.maxnum.f32(float %660, float %667)
  %669 = fsub contract float %565, %668
  %670 = fmul contract float %669, 0x3FF7154760000000
  %671 = tail call contract float @llvm.amdgcn.exp2.f32(float %670)
  %672 = select contract i1 %564, float %671, float 0.000000e+00
  %673 = insertelement <16 x float> %555, float %672, i64 0
  %674 = shufflevector <16 x float> %673, <16 x float> %563, <16 x i32> <i32 0, i32 17, i32 18, i32 19, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %675 = bitcast <16 x float> %674 to <4 x i128>
  %676 = extractelement <4 x i128> %675, i64 0
  %677 = bitcast i128 %676 to <4 x float>
  %678 = extractelement <4 x float> %677, i64 1
  %679 = fsub contract float %678, %668
  %680 = fmul contract float %679, 0x3FF7154760000000
  %681 = tail call contract float @llvm.amdgcn.exp2.f32(float %680)
  %682 = select contract i1 %569, float %681, float 0.000000e+00
  %683 = shufflevector <4 x float> %677, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %684 = shufflevector <4 x float> %677, <4 x float> poison, <16 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %685 = shufflevector <16 x float> %684, <16 x float> %674, <16 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %686 = insertelement <16 x float> %685, float %682, i64 1
  %687 = shufflevector <16 x float> %686, <16 x float> %683, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %688 = fadd contract float %672, %682
  %689 = bitcast <16 x float> %687 to <4 x i128>
  %690 = extractelement <4 x i128> %689, i64 0
  %691 = bitcast i128 %690 to <4 x float>
  %692 = extractelement <4 x float> %691, i64 2
  %693 = fsub contract float %692, %668
  %694 = fmul contract float %693, 0x3FF7154760000000
  %695 = tail call contract float @llvm.amdgcn.exp2.f32(float %694)
  %696 = select contract i1 %574, float %695, float 0.000000e+00
  %697 = shufflevector <4 x float> %691, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %698 = shufflevector <4 x float> %691, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %699 = shufflevector <16 x float> %698, <16 x float> %687, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %700 = insertelement <16 x float> %699, float %696, i64 2
  %701 = shufflevector <16 x float> %700, <16 x float> %697, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %702 = fadd contract float %688, %696
  %703 = bitcast <16 x float> %701 to <4 x i128>
  %704 = extractelement <4 x i128> %703, i64 0
  %705 = bitcast i128 %704 to <4 x float>
  %706 = extractelement <4 x float> %705, i64 3
  %707 = fsub contract float %706, %668
  %708 = fmul contract float %707, 0x3FF7154760000000
  %709 = tail call contract float @llvm.amdgcn.exp2.f32(float %708)
  %710 = select contract i1 %579, float %709, float 0.000000e+00
  %711 = shufflevector <4 x float> %705, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %712 = shufflevector <16 x float> %711, <16 x float> %701, <16 x i32> <i32 0, i32 1, i32 2, i32 poison, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %713 = insertelement <16 x float> %712, float %710, i64 3
  %714 = fadd contract float %702, %710
  %715 = bitcast <16 x float> %713 to <4 x i128>
  %716 = extractelement <4 x i128> %715, i64 1
  %717 = bitcast i128 %716 to <4 x float>
  %718 = extractelement <4 x float> %717, i64 0
  %719 = fsub contract float %718, %668
  %720 = fmul contract float %719, 0x3FF7154760000000
  %721 = tail call contract float @llvm.amdgcn.exp2.f32(float %720)
  %722 = select contract i1 %586, float %721, float 0.000000e+00
  %723 = shufflevector <4 x float> %717, <4 x float> poison, <16 x i32> <i32 poison, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %724 = insertelement <16 x float> %713, float %722, i64 4
  %725 = shufflevector <16 x float> %724, <16 x float> %723, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %726 = fadd contract float %714, %722
  %727 = bitcast <16 x float> %725 to <4 x i128>
  %728 = extractelement <4 x i128> %727, i64 1
  %729 = bitcast i128 %728 to <4 x float>
  %730 = extractelement <4 x float> %729, i64 1
  %731 = fsub contract float %730, %668
  %732 = fmul contract float %731, 0x3FF7154760000000
  %733 = tail call contract float @llvm.amdgcn.exp2.f32(float %732)
  %734 = select contract i1 %591, float %733, float 0.000000e+00
  %735 = shufflevector <4 x float> %729, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %736 = shufflevector <4 x float> %729, <4 x float> poison, <16 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %737 = shufflevector <16 x float> %725, <16 x float> %736, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %738 = insertelement <16 x float> %737, float %734, i64 5
  %739 = shufflevector <16 x float> %738, <16 x float> %735, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %740 = fadd contract float %726, %734
  %741 = bitcast <16 x float> %739 to <4 x i128>
  %742 = extractelement <4 x i128> %741, i64 1
  %743 = bitcast i128 %742 to <4 x float>
  %744 = extractelement <4 x float> %743, i64 2
  %745 = fsub contract float %744, %668
  %746 = fmul contract float %745, 0x3FF7154760000000
  %747 = tail call contract float @llvm.amdgcn.exp2.f32(float %746)
  %748 = select contract i1 %596, float %747, float 0.000000e+00
  %749 = shufflevector <4 x float> %743, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %750 = shufflevector <4 x float> %743, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %751 = shufflevector <16 x float> %739, <16 x float> %750, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %752 = insertelement <16 x float> %751, float %748, i64 6
  %753 = shufflevector <16 x float> %752, <16 x float> %749, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %754 = fadd contract float %740, %748
  %755 = bitcast <16 x float> %753 to <4 x i128>
  %756 = extractelement <4 x i128> %755, i64 1
  %757 = bitcast i128 %756 to <4 x float>
  %758 = extractelement <4 x float> %757, i64 3
  %759 = fsub contract float %758, %668
  %760 = fmul contract float %759, 0x3FF7154760000000
  %761 = tail call contract float @llvm.amdgcn.exp2.f32(float %760)
  %762 = select contract i1 %601, float %761, float 0.000000e+00
  %763 = shufflevector <4 x float> %757, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %764 = shufflevector <16 x float> %753, <16 x float> %763, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 poison, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %765 = insertelement <16 x float> %764, float %762, i64 7
  %766 = fadd contract float %754, %762
  %767 = bitcast <16 x float> %765 to <4 x i128>
  %768 = extractelement <4 x i128> %767, i64 2
  %769 = bitcast i128 %768 to <4 x float>
  %770 = extractelement <4 x float> %769, i64 0
  %771 = fsub contract float %770, %668
  %772 = fmul contract float %771, 0x3FF7154760000000
  %773 = tail call contract float @llvm.amdgcn.exp2.f32(float %772)
  %774 = select contract i1 %608, float %773, float 0.000000e+00
  %775 = shufflevector <4 x float> %769, <4 x float> poison, <16 x i32> <i32 poison, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %776 = insertelement <16 x float> %765, float %774, i64 8
  %777 = shufflevector <16 x float> %776, <16 x float> %775, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 17, i32 18, i32 19, i32 12, i32 13, i32 14, i32 15>
  %778 = fadd contract float %766, %774
  %779 = bitcast <16 x float> %777 to <4 x i128>
  %780 = extractelement <4 x i128> %779, i64 2
  %781 = bitcast i128 %780 to <4 x float>
  %782 = extractelement <4 x float> %781, i64 1
  %783 = fsub contract float %782, %668
  %784 = fmul contract float %783, 0x3FF7154760000000
  %785 = tail call contract float @llvm.amdgcn.exp2.f32(float %784)
  %786 = select contract i1 %613, float %785, float 0.000000e+00
  %787 = shufflevector <4 x float> %781, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %788 = shufflevector <4 x float> %781, <4 x float> poison, <16 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %789 = shufflevector <16 x float> %777, <16 x float> %788, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 poison, i32 poison, i32 poison, i32 12, i32 13, i32 14, i32 15>
  %790 = insertelement <16 x float> %789, float %786, i64 9
  %791 = shufflevector <16 x float> %790, <16 x float> %787, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 18, i32 19, i32 12, i32 13, i32 14, i32 15>
  %792 = fadd contract float %778, %786
  %793 = bitcast <16 x float> %791 to <4 x i128>
  %794 = extractelement <4 x i128> %793, i64 2
  %795 = bitcast i128 %794 to <4 x float>
  %796 = extractelement <4 x float> %795, i64 2
  %797 = fsub contract float %796, %668
  %798 = fmul contract float %797, 0x3FF7154760000000
  %799 = tail call contract float @llvm.amdgcn.exp2.f32(float %798)
  %800 = select contract i1 %618, float %799, float 0.000000e+00
  %801 = shufflevector <4 x float> %795, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %802 = shufflevector <4 x float> %795, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %803 = shufflevector <16 x float> %791, <16 x float> %802, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 poison, i32 poison, i32 12, i32 13, i32 14, i32 15>
  %804 = insertelement <16 x float> %803, float %800, i64 10
  %805 = shufflevector <16 x float> %804, <16 x float> %801, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 19, i32 12, i32 13, i32 14, i32 15>
  %806 = fadd contract float %792, %800
  %807 = bitcast <16 x float> %805 to <4 x i128>
  %808 = extractelement <4 x i128> %807, i64 2
  %809 = bitcast i128 %808 to <4 x float>
  %810 = extractelement <4 x float> %809, i64 3
  %811 = fsub contract float %810, %668
  %812 = fmul contract float %811, 0x3FF7154760000000
  %813 = tail call contract float @llvm.amdgcn.exp2.f32(float %812)
  %814 = select contract i1 %623, float %813, float 0.000000e+00
  %815 = shufflevector <4 x float> %809, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %816 = shufflevector <16 x float> %805, <16 x float> %815, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 poison, i32 12, i32 13, i32 14, i32 15>
  %817 = insertelement <16 x float> %816, float %814, i64 11
  %818 = fadd contract float %806, %814
  %819 = bitcast <16 x float> %817 to <4 x i128>
  %820 = extractelement <4 x i128> %819, i64 3
  %821 = bitcast i128 %820 to <4 x float>
  %822 = extractelement <4 x float> %821, i64 0
  %823 = fsub contract float %822, %668
  %824 = fmul contract float %823, 0x3FF7154760000000
  %825 = tail call contract float @llvm.amdgcn.exp2.f32(float %824)
  %826 = select contract i1 %630, float %825, float 0.000000e+00
  %827 = shufflevector <4 x float> %821, <4 x float> poison, <16 x i32> <i32 poison, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %828 = insertelement <16 x float> %817, float %826, i64 12
  %829 = shufflevector <16 x float> %828, <16 x float> %827, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 17, i32 18, i32 19>
  %830 = fadd contract float %818, %826
  %831 = bitcast <16 x float> %829 to <4 x i128>
  %832 = extractelement <4 x i128> %831, i64 3
  %833 = bitcast i128 %832 to <4 x float>
  %834 = extractelement <4 x float> %833, i64 1
  %835 = fsub contract float %834, %668
  %836 = fmul contract float %835, 0x3FF7154760000000
  %837 = tail call contract float @llvm.amdgcn.exp2.f32(float %836)
  %838 = select contract i1 %635, float %837, float 0.000000e+00
  %839 = shufflevector <4 x float> %833, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %840 = shufflevector <4 x float> %833, <4 x float> poison, <16 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %841 = shufflevector <16 x float> %829, <16 x float> %840, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 poison, i32 poison, i32 poison>
  %842 = insertelement <16 x float> %841, float %838, i64 13
  %843 = shufflevector <16 x float> %842, <16 x float> %839, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 18, i32 19>
  %844 = fadd contract float %830, %838
  %845 = bitcast <16 x float> %843 to <4 x i128>
  %846 = extractelement <4 x i128> %845, i64 3
  %847 = bitcast i128 %846 to <4 x float>
  %848 = extractelement <4 x float> %847, i64 2
  %849 = fsub contract float %848, %668
  %850 = fmul contract float %849, 0x3FF7154760000000
  %851 = tail call contract float @llvm.amdgcn.exp2.f32(float %850)
  %852 = select contract i1 %640, float %851, float 0.000000e+00
  %853 = shufflevector <4 x float> %847, <4 x float> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %854 = shufflevector <4 x float> %847, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %855 = shufflevector <16 x float> %843, <16 x float> %854, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 poison, i32 poison>
  %856 = insertelement <16 x float> %855, float %852, i64 14
  %857 = shufflevector <16 x float> %856, <16 x float> %853, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 19>
  %858 = fadd contract float %844, %852
  %859 = bitcast <16 x float> %857 to <4 x i128>
  %860 = extractelement <4 x i128> %859, i64 3
  %861 = bitcast i128 %860 to <4 x float>
  %862 = extractelement <4 x float> %861, i64 3
  %863 = fsub contract float %862, %668
  %864 = fmul contract float %863, 0x3FF7154760000000
  %865 = tail call contract float @llvm.amdgcn.exp2.f32(float %864)
  %866 = select contract i1 %645, float %865, float 0.000000e+00
  %867 = shufflevector <4 x float> %861, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %868 = shufflevector <16 x float> %857, <16 x float> %867, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 poison>
  %869 = fadd contract float %858, %866
  %870 = bitcast float %869 to i32
  %871 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %657, i32 %870)
  %872 = bitcast i32 %871 to float
  %873 = fadd contract float %869, %872
  %874 = bitcast float %873 to i32
  %875 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %665, i32 %874)
  %876 = insertelement <16 x float> %868, float %866, i64 15
  fence syncscope("workgroup") release
  call void @etx_imp_slot_barrier(i32 %etx.slot, i32 4)
  fence syncscope("workgroup") acquire
  %877 = icmp samesign ult i32 %25, 16
  br i1 %877, label %878, label %884

878:                                              ; preds = %123
  %879 = bitcast i32 %875 to float
  %880 = fadd contract float %873, %879
  %881 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @etx_lds_arena, i32 0), i32 %556
  %882 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %881, i32 %26
  %883 = getelementptr inbounds nuw i8, ptr addrspace(3) %882, i32 256
  store float %668, ptr addrspace(3) %882, align 4, !tbaa !9
  store float %880, ptr addrspace(3) %883, align 4, !tbaa !9
  br label %884

884:                                              ; preds = %878, %123
  fence syncscope("workgroup") release
  call void @etx_imp_slot_barrier(i32 %etx.slot, i32 4)
  fence syncscope("workgroup") acquire
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %21) #10
  %885 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @etx_lds_arena, i32 0), i32 %26
  %886 = load float, ptr addrspace(3) %885, align 4, !tbaa !9
  %887 = tail call nsz contract noundef float @llvm.maxnum.f32(float %886, float 0xC7EFFFFFE0000000)
  %888 = getelementptr inbounds nuw i8, ptr addrspace(3) %885, i32 64
  %889 = load float, ptr addrspace(3) %888, align 4, !tbaa !9
  %890 = getelementptr inbounds nuw i8, ptr addrspace(5) %21, i32 4
  %891 = tail call nsz contract noundef float @llvm.maxnum.f32(float %887, float %889)
  %892 = getelementptr inbounds nuw i8, ptr addrspace(3) %885, i32 128
  %893 = load float, ptr addrspace(3) %892, align 4, !tbaa !9
  %894 = getelementptr inbounds nuw i8, ptr addrspace(5) %21, i32 8
  %895 = tail call nsz contract noundef float @llvm.maxnum.f32(float %891, float %893)
  %896 = getelementptr inbounds nuw i8, ptr addrspace(3) %885, i32 192
  %897 = load float, ptr addrspace(3) %896, align 4, !tbaa !9
  %898 = getelementptr inbounds nuw i8, ptr addrspace(5) %21, i32 12
  %899 = tail call nsz contract noundef float @llvm.maxnum.f32(float %895, float %897)
  %900 = getelementptr inbounds nuw i8, ptr addrspace(3) %885, i32 256
  %901 = load float, ptr addrspace(3) %900, align 4, !tbaa !9
  %902 = insertelement <2 x float> poison, float %886, i64 0
  %903 = insertelement <2 x float> %902, float %889, i64 1
  %904 = insertelement <2 x float> poison, float %899, i64 0
  %905 = shufflevector <2 x float> %904, <2 x float> poison, <2 x i32> zeroinitializer
  %906 = fsub contract <2 x float> %903, %905
  %907 = fmul contract <2 x float> %906, splat (float 0x3FF7154760000000)
  %908 = extractelement <2 x float> %907, i64 0
  %909 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %908)
  store float %909, ptr addrspace(5) %21, align 16, !tbaa !9
  %910 = fmul contract float %901, %909
  %911 = fadd contract float %910, 0.000000e+00
  %912 = extractelement <2 x float> %907, i64 1
  %913 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %912)
  store float %913, ptr addrspace(5) %890, align 4, !tbaa !9
  %914 = getelementptr inbounds nuw i8, ptr addrspace(3) %885, i32 320
  %915 = load float, ptr addrspace(3) %914, align 4, !tbaa !9
  %916 = fmul contract float %915, %913
  %917 = fadd contract float %911, %916
  %918 = getelementptr inbounds nuw i8, ptr addrspace(3) %885, i32 384
  %919 = load float, ptr addrspace(3) %918, align 4, !tbaa !9
  %920 = insertelement <2 x float> poison, float %893, i64 0
  %921 = insertelement <2 x float> %920, float %897, i64 1
  %922 = fsub contract <2 x float> %921, %905
  %923 = fmul contract <2 x float> %922, splat (float 0x3FF7154760000000)
  %924 = extractelement <2 x float> %923, i64 0
  %925 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %924)
  store float %925, ptr addrspace(5) %894, align 8, !tbaa !9
  %926 = fmul contract float %919, %925
  %927 = fadd contract float %917, %926
  %928 = extractelement <2 x float> %923, i64 1
  %929 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %928)
  store float %929, ptr addrspace(5) %898, align 4, !tbaa !9
  %930 = getelementptr inbounds nuw i8, ptr addrspace(3) %885, i32 448
  %931 = load float, ptr addrspace(3) %930, align 4, !tbaa !9
  %932 = fmul contract float %931, %929
  %933 = fadd contract float %927, %932
  %934 = fadd contract float %933, 0x3EB0C6F7A0000000
  %935 = fdiv contract float 1.000000e+00, %934
  %936 = getelementptr inbounds nuw [4 x i8], ptr addrspace(5) %21, i32 %24
  %937 = load float, ptr addrspace(5) %936, align 4, !tbaa !9
  %938 = fmul contract float %935, %937
  fence syncscope("workgroup") release
  call void @etx_imp_slot_barrier(i32 %etx.slot, i32 4)
  fence syncscope("workgroup") acquire
  %939 = insertelement <4 x float> poison, float %938, i64 0
  %940 = shufflevector <4 x float> %939, <4 x float> poison, <4 x i32> zeroinitializer
  %941 = getelementptr inbounds nuw [2048 x i8], ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @etx_lds_arena, i32 0), i32 %24
  %942 = getelementptr inbounds nuw [32 x i8], ptr addrspace(3) %941, i32 %26
  %943 = getelementptr inbounds nuw [8 x i8], ptr addrspace(3) %942, i32 %27
  %944 = bitcast <16 x float> %876 to <4 x i128>
  %945 = extractelement <4 x i128> %944, i64 0
  %946 = bitcast i128 %945 to <4 x float>
  %947 = fmul contract <4 x float> %940, %946
  %948 = shufflevector <4 x float> %947, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %949 = shufflevector <16 x float> %948, <16 x float> %876, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %950 = bitcast <16 x float> %949 to <4 x i128>
  %951 = extractelement <4 x i128> %950, i64 0
  %952 = bitcast i128 %951 to <4 x i32>
  %953 = add <4 x i32> %952, splat (i32 32767)
  %954 = lshr <4 x i32> %952, splat (i32 16)
  %955 = and <4 x i32> %954, splat (i32 1)
  %956 = add <4 x i32> %953, %955
  %957 = lshr <4 x i32> %956, splat (i32 16)
  %958 = trunc nuw <4 x i32> %957 to <4 x i16>
  store <4 x i16> %958, ptr addrspace(3) %943, align 8, !tbaa !8
  %959 = extractelement <4 x i128> %950, i64 1
  %960 = bitcast i128 %959 to <4 x float>
  %961 = fmul contract <4 x float> %940, %960
  %962 = shufflevector <4 x float> %961, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %963 = shufflevector <16 x float> %949, <16 x float> %962, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %964 = bitcast <16 x float> %963 to <4 x i128>
  %965 = extractelement <4 x i128> %964, i64 1
  %966 = bitcast i128 %965 to <4 x i32>
  %967 = add <4 x i32> %966, splat (i32 32767)
  %968 = lshr <4 x i32> %966, splat (i32 16)
  %969 = and <4 x i32> %968, splat (i32 1)
  %970 = add <4 x i32> %967, %969
  %971 = lshr <4 x i32> %970, splat (i32 16)
  %972 = trunc nuw <4 x i32> %971 to <4 x i16>
  %973 = getelementptr inbounds nuw i8, ptr addrspace(3) %943, i32 512
  store <4 x i16> %972, ptr addrspace(3) %973, align 8, !tbaa !8
  %974 = extractelement <4 x i128> %964, i64 2
  %975 = bitcast i128 %974 to <4 x float>
  %976 = fmul contract <4 x float> %940, %975
  %977 = shufflevector <4 x float> %976, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %978 = shufflevector <16 x float> %963, <16 x float> %977, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 18, i32 19, i32 12, i32 13, i32 14, i32 15>
  %979 = bitcast <16 x float> %978 to <4 x i128>
  %980 = extractelement <4 x i128> %979, i64 2
  %981 = bitcast i128 %980 to <4 x i32>
  %982 = add <4 x i32> %981, splat (i32 32767)
  %983 = lshr <4 x i32> %981, splat (i32 16)
  %984 = and <4 x i32> %983, splat (i32 1)
  %985 = add <4 x i32> %982, %984
  %986 = lshr <4 x i32> %985, splat (i32 16)
  %987 = trunc nuw <4 x i32> %986 to <4 x i16>
  %988 = getelementptr inbounds nuw i8, ptr addrspace(3) %943, i32 1024
  store <4 x i16> %987, ptr addrspace(3) %988, align 8, !tbaa !8
  %989 = extractelement <4 x i128> %979, i64 3
  %990 = bitcast i128 %989 to <4 x float>
  %991 = fmul contract <4 x float> %940, %990
  %992 = bitcast <4 x float> %991 to <4 x i32>
  %993 = add <4 x i32> %992, splat (i32 32767)
  %994 = lshr <4 x i32> %992, splat (i32 16)
  %995 = and <4 x i32> %994, splat (i32 1)
  %996 = add <4 x i32> %993, %995
  %997 = lshr <4 x i32> %996, splat (i32 16)
  %998 = trunc nuw <4 x i32> %997 to <4 x i16>
  %999 = getelementptr inbounds nuw i8, ptr addrspace(3) %943, i32 1536
  store <4 x i16> %998, ptr addrspace(3) %999, align 8, !tbaa !8
  %1000 = zext i32 %etx.lx to i64
  %1001 = and i64 %1000, 4294967288
  %1002 = icmp eq i64 %1001, 0
  br i1 %1002, label %1003, label %1016

1003:                                             ; preds = %884
  %1004 = zext i32 %93 to i64
  %1005 = mul nuw nsw i64 %44, %1004
  %1006 = and i64 %47, 4294967295
  %1007 = zext i32 %92 to i64
  %1008 = and i64 %22, 15
  %1009 = add nuw nsw i64 %1008, %1007
  %1010 = add nuw i64 %1009, %1005
  %1011 = mul nuw i64 %1010, %1006
  %1012 = and i64 %45, 4294967295
  %1013 = add nuw nsw i64 %1011, %1012
  %1014 = getelementptr inbounds [4 x i8], ptr addrspace(1) %14, i64 %1013
  store float %899, ptr addrspace(1) %1014, align 4, !tbaa !9
  %1015 = getelementptr inbounds [4 x i8], ptr addrspace(1) %13, i64 %1013
  store float %933, ptr addrspace(1) %1015, align 4, !tbaa !9
  br label %1016

1016:                                             ; preds = %1003, %884
  fence syncscope("workgroup") release
  call void @etx_imp_slot_barrier(i32 %etx.slot, i32 4)
  fence syncscope("workgroup") acquire
  %1017 = getelementptr inbounds nuw [32 x i8], ptr addrspace(3) %124, i32 %26
  %1018 = load <4 x i16>, ptr addrspace(3) %1017, align 16, !tbaa !8
  %1019 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %273, <4 x i16> %1018, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %1020 = getelementptr inbounds nuw i8, ptr addrspace(3) %1017, i32 8
  %1021 = load <4 x i16>, ptr addrspace(3) %1020, align 8, !tbaa !8
  %1022 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %275, <4 x i16> %1021, <4 x float> %1019, i32 0, i32 0, i32 0)
  %1023 = getelementptr inbounds nuw i8, ptr addrspace(3) %1017, i32 16
  %1024 = load <4 x i16>, ptr addrspace(3) %1023, align 16, !tbaa !8
  %1025 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %277, <4 x i16> %1024, <4 x float> %1022, i32 0, i32 0, i32 0)
  %1026 = getelementptr inbounds nuw i8, ptr addrspace(3) %1017, i32 24
  %1027 = load <4 x i16>, ptr addrspace(3) %1026, align 8, !tbaa !8
  %1028 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %279, <4 x i16> %1027, <4 x float> %1025, i32 0, i32 0, i32 0)
  %1029 = getelementptr inbounds nuw i8, ptr addrspace(3) %1017, i32 2048
  %1030 = load <4 x i16>, ptr addrspace(3) %1029, align 16, !tbaa !8
  %1031 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %283, <4 x i16> %1030, <4 x float> %1028, i32 0, i32 0, i32 0)
  %1032 = getelementptr inbounds nuw i8, ptr addrspace(3) %1017, i32 2056
  %1033 = load <4 x i16>, ptr addrspace(3) %1032, align 8, !tbaa !8
  %1034 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %285, <4 x i16> %1033, <4 x float> %1031, i32 0, i32 0, i32 0)
  %1035 = getelementptr inbounds nuw i8, ptr addrspace(3) %1017, i32 2064
  %1036 = load <4 x i16>, ptr addrspace(3) %1035, align 16, !tbaa !8
  %1037 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %287, <4 x i16> %1036, <4 x float> %1034, i32 0, i32 0, i32 0)
  %1038 = getelementptr inbounds nuw i8, ptr addrspace(3) %1017, i32 2072
  %1039 = load <4 x i16>, ptr addrspace(3) %1038, align 8, !tbaa !8
  %1040 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %289, <4 x i16> %1039, <4 x float> %1037, i32 0, i32 0, i32 0)
  %1041 = getelementptr inbounds nuw i8, ptr addrspace(3) %1017, i32 4096
  %1042 = load <4 x i16>, ptr addrspace(3) %1041, align 16, !tbaa !8
  %1043 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %293, <4 x i16> %1042, <4 x float> %1040, i32 0, i32 0, i32 0)
  %1044 = getelementptr inbounds nuw i8, ptr addrspace(3) %1017, i32 4104
  %1045 = load <4 x i16>, ptr addrspace(3) %1044, align 8, !tbaa !8
  %1046 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %295, <4 x i16> %1045, <4 x float> %1043, i32 0, i32 0, i32 0)
  %1047 = getelementptr inbounds nuw i8, ptr addrspace(3) %1017, i32 4112
  %1048 = load <4 x i16>, ptr addrspace(3) %1047, align 16, !tbaa !8
  %1049 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %297, <4 x i16> %1048, <4 x float> %1046, i32 0, i32 0, i32 0)
  %1050 = getelementptr inbounds nuw i8, ptr addrspace(3) %1017, i32 4120
  %1051 = load <4 x i16>, ptr addrspace(3) %1050, align 8, !tbaa !8
  %1052 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %299, <4 x i16> %1051, <4 x float> %1049, i32 0, i32 0, i32 0)
  %1053 = getelementptr inbounds nuw i8, ptr addrspace(3) %1017, i32 6144
  %1054 = load <4 x i16>, ptr addrspace(3) %1053, align 16, !tbaa !8
  %1055 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %303, <4 x i16> %1054, <4 x float> %1052, i32 0, i32 0, i32 0)
  %1056 = getelementptr inbounds nuw i8, ptr addrspace(3) %1017, i32 6152
  %1057 = load <4 x i16>, ptr addrspace(3) %1056, align 8, !tbaa !8
  %1058 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %305, <4 x i16> %1057, <4 x float> %1055, i32 0, i32 0, i32 0)
  %1059 = getelementptr inbounds nuw i8, ptr addrspace(3) %1017, i32 6160
  %1060 = load <4 x i16>, ptr addrspace(3) %1059, align 16, !tbaa !8
  %1061 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %307, <4 x i16> %1060, <4 x float> %1058, i32 0, i32 0, i32 0)
  %1062 = getelementptr inbounds nuw i8, ptr addrspace(3) %1017, i32 6168
  %1063 = load <4 x i16>, ptr addrspace(3) %1062, align 8, !tbaa !8
  %1064 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %309, <4 x i16> %1063, <4 x float> %1061, i32 0, i32 0, i32 0)
  %1065 = bitcast <4 x float> %1064 to <4 x i32>
  %1066 = add <4 x i32> %1065, splat (i32 32767)
  %1067 = lshr <4 x i32> %1065, splat (i32 16)
  %1068 = and <4 x i32> %1067, splat (i32 1)
  %1069 = add <4 x i32> %1066, %1068
  %1070 = lshr <4 x i32> %1069, splat (i32 16)
  %1071 = trunc nuw <4 x i32> %1070 to <4 x i16>
  %1072 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %314, <4 x i16> %1018, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %1073 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %316, <4 x i16> %1021, <4 x float> %1072, i32 0, i32 0, i32 0)
  %1074 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %318, <4 x i16> %1024, <4 x float> %1073, i32 0, i32 0, i32 0)
  %1075 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %320, <4 x i16> %1027, <4 x float> %1074, i32 0, i32 0, i32 0)
  %1076 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %322, <4 x i16> %1030, <4 x float> %1075, i32 0, i32 0, i32 0)
  %1077 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %324, <4 x i16> %1033, <4 x float> %1076, i32 0, i32 0, i32 0)
  %1078 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %326, <4 x i16> %1036, <4 x float> %1077, i32 0, i32 0, i32 0)
  %1079 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %328, <4 x i16> %1039, <4 x float> %1078, i32 0, i32 0, i32 0)
  %1080 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %330, <4 x i16> %1042, <4 x float> %1079, i32 0, i32 0, i32 0)
  %1081 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %332, <4 x i16> %1045, <4 x float> %1080, i32 0, i32 0, i32 0)
  %1082 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %334, <4 x i16> %1048, <4 x float> %1081, i32 0, i32 0, i32 0)
  %1083 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %336, <4 x i16> %1051, <4 x float> %1082, i32 0, i32 0, i32 0)
  %1084 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %338, <4 x i16> %1054, <4 x float> %1083, i32 0, i32 0, i32 0)
  %1085 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %340, <4 x i16> %1057, <4 x float> %1084, i32 0, i32 0, i32 0)
  %1086 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %342, <4 x i16> %1060, <4 x float> %1085, i32 0, i32 0, i32 0)
  %1087 = tail call contract noundef <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %344, <4 x i16> %1063, <4 x float> %1086, i32 0, i32 0, i32 0)
  %1088 = bitcast <4 x float> %1087 to <4 x i32>
  %1089 = add <4 x i32> %1088, splat (i32 32767)
  %1090 = lshr <4 x i32> %1088, splat (i32 16)
  %1091 = and <4 x i32> %1090, splat (i32 1)
  %1092 = add <4 x i32> %1089, %1091
  %1093 = lshr <4 x i32> %1092, splat (i32 16)
  %1094 = trunc nuw <4 x i32> %1093 to <4 x i16>
  fence syncscope("workgroup") release
  call void @etx_imp_slot_barrier(i32 %etx.slot, i32 4)
  fence syncscope("workgroup") acquire
  store <4 x i16> %1071, ptr addrspace(3) %943, align 8, !tbaa !8
  store <4 x i16> %1094, ptr addrspace(3) %973, align 8, !tbaa !8
  fence syncscope("workgroup") release
  call void @etx_imp_slot_barrier(i32 %etx.slot, i32 4)
  fence syncscope("workgroup") acquire
  %1095 = icmp eq i32 %24, 0
  br i1 %1095, label %1096, label %1134

1096:                                             ; preds = %1016
  %1097 = lshr i32 %26, 3
  %1098 = shl nuw nsw i32 %23, 1
  %1099 = and i32 %1098, 2
  %1100 = lshr i32 %23, 1
  %1101 = and i32 %1100, 3
  %1102 = getelementptr inbounds nuw [2048 x i8], ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @etx_lds_arena, i32 0), i32 %1101
  %1103 = getelementptr inbounds nuw [512 x i8], ptr addrspace(3) %1102, i32 %1097
  %1104 = getelementptr inbounds nuw [32 x i8], ptr addrspace(3) %1103, i32 %27
  %1105 = getelementptr inbounds nuw [8 x i8], ptr addrspace(3) %1104, i32 %1099
  %1106 = load <4 x i16>, ptr addrspace(3) %1105, align 16, !tbaa !8
  %1107 = getelementptr inbounds nuw i8, ptr addrspace(3) %1105, i32 8
  %1108 = load <4 x i16>, ptr addrspace(3) %1107, align 8, !tbaa !8
  %1109 = getelementptr inbounds nuw i8, ptr addrspace(3) %1105, i32 128
  %1110 = load <4 x i16>, ptr addrspace(3) %1109, align 16, !tbaa !8
  %1111 = getelementptr inbounds nuw i8, ptr addrspace(3) %1105, i32 136
  %1112 = load <4 x i16>, ptr addrspace(3) %1111, align 8, !tbaa !8
  %1113 = shl i64 %47, 7
  %1114 = and i64 %1113, 4294967168
  %1115 = mul i32 %93, %29
  %1116 = zext i32 %1115 to i64
  %1117 = mul nuw nsw i64 %1114, %1116
  %1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %15, i64 %1117
  %1119 = shl i64 %45, 7
  %1120 = and i64 %1119, 4294967168
  %1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1118, i64 %1120
  %1122 = or disjoint i32 %27, %92
  %1123 = zext nneg i32 %102 to i64
  %1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1121, i64 %1123
  %1125 = zext i32 %1122 to i64
  %1126 = mul nuw nsw i64 %1114, %1125
  %1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1124, i64 %1126
  store <4 x i16> %1106, ptr addrspace(1) %1127, align 8
  %1128 = getelementptr inbounds nuw i8, ptr addrspace(1) %1127, i64 8
  store <4 x i16> %1108, ptr addrspace(1) %1128, align 8, !tbaa !8
  %1129 = or disjoint i32 %1122, 4
  %1130 = zext i32 %1129 to i64
  %1131 = mul nuw nsw i64 %1114, %1130
  %1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1124, i64 %1131
  store <4 x i16> %1110, ptr addrspace(1) %1132, align 8
  %1133 = getelementptr inbounds nuw i8, ptr addrspace(1) %1132, i64 8
  store <4 x i16> %1112, ptr addrspace(1) %1133, align 8, !tbaa !8
  br label %1134

1134:                                             ; preds = %1096, %1016
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %21) #10
  br label %1135

1135:                                             ; preds = %1134, %43, %33
  ret void
}

define void @etx_imp_attn_mfma16_g8(ptr noundef %args, i32 noundef %blk, i32 noundef %tid, i32 noundef %slot) #12 {
  %args4 = addrspacecast ptr %args to ptr addrspace(4)
  %a0.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 0
  %a0 = load ptr addrspace(1), ptr addrspace(4) %a0.p, align 8
  %a1.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 8
  %a1 = load ptr addrspace(1), ptr addrspace(4) %a1.p, align 8
  %a2.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 16
  %a2 = load ptr addrspace(1), ptr addrspace(4) %a2.p, align 8
  %a3.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 24
  %a3 = load i32, ptr addrspace(4) %a3.p, align 4
  %a4.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 28
  %a4 = load float, ptr addrspace(4) %a4.p, align 4
  %a5.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 32
  %a5 = load ptr addrspace(1), ptr addrspace(4) %a5.p, align 8
  %a6.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 40
  %a6 = load ptr addrspace(1), ptr addrspace(4) %a6.p, align 8
  %a7.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 48
  %a7 = load ptr addrspace(1), ptr addrspace(4) %a7.p, align 8
  %a8.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 56
  %a8 = load i32, ptr addrspace(4) %a8.p, align 4
  %a9.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 64
  %a9 = load ptr addrspace(1), ptr addrspace(4) %a9.p, align 8
  %a10.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 72
  %a10 = load i32, ptr addrspace(4) %a10.p, align 4
  %a11.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 76
  %a11 = load i32, ptr addrspace(4) %a11.p, align 4
  %a12.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 80
  %a12 = load i32, ptr addrspace(4) %a12.p, align 4
  %a13.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 88
  %a13 = load ptr addrspace(1), ptr addrspace(4) %a13.p, align 8
  %a14.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 96
  %a14 = load ptr addrspace(1), ptr addrspace(4) %a14.p, align 8
  %a15.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 104
  %a15 = load ptr addrspace(1), ptr addrspace(4) %a15.p, align 8
  %a16.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 112
  %a16 = load ptr addrspace(1), ptr addrspace(4) %a16.p, align 8
  %a17.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 120
  %a17 = load i32, ptr addrspace(4) %a17.p, align 4
  %a18.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 128
  %a18 = load ptr addrspace(1), ptr addrspace(4) %a18.p, align 8
  %a19.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 136
  %a19 = load ptr addrspace(1), ptr addrspace(4) %a19.p, align 8
  %gx.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 144
  %gx = load i32, ptr addrspace(4) %gx.p, align 4
  %gy.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 148
  %gy = load i32, ptr addrspace(4) %gy.p, align 4
  %gz.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 152
  %gz = load i32, ptr addrspace(4) %gz.p, align 4
  %bx = urem i32 %blk, %gx
  %b1 = udiv i32 %blk, %gx
  %by = urem i32 %b1, %gy
  %bz = udiv i32 %b1, %gy
  call void @etx_imp_attn_mfma16_g8.body.s0(ptr addrspace(1) %a0, ptr addrspace(1) %a1, ptr addrspace(1) %a2, i32 %a3, float %a4, ptr addrspace(1) %a5, ptr addrspace(1) %a6, ptr addrspace(1) %a7, i32 %a8, ptr addrspace(1) %a9, i32 %a10, i32 %a11, i32 %a12, ptr addrspace(1) %a13, ptr addrspace(1) %a14, ptr addrspace(1) %a15, ptr addrspace(1) %a16, i32 %a17, ptr addrspace(1) %a18, ptr addrspace(1) %a19, i32 %bx, i32 %by, i32 %bz, i32 %gx, i32 %gy, i32 %gz, i32 %tid, i32 %slot)
  ret void
}
attributes #11 = { alwaysinline convergent mustprogress nofree norecurse nounwind willreturn "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx942" }
attributes #12 = { alwaysinline convergent nounwind "target-cpu"="gfx942" }
