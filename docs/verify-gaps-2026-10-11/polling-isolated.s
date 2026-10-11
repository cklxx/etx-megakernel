	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx942"
	.amdhsa_code_object_version 6
	.text
	.protected	k_polling_wave0         ; -- Begin function k_polling_wave0
	.globl	k_polling_wave0
	.p2align	8
	.type	k_polling_wave0,@function
k_polling_wave0:                        ; @k_polling_wave0
.Lfunc_begin0:
	.file	0 "/Users/ckl/code/.wt/etx-verifygaps" "/tmp/etx-verifygaps/polling/poll.hip" md5 0x5dd15dfd5ce497cd340c788a220b7012
	.loc	0 7 0                           ; /tmp/etx-verifygaps/polling/poll.hip:7:0
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_dwordx2 s[2:3], s[0:1], 0x20
	v_mov_b32_e32 v0, 0
.Ltmp0:
	.file	1 "etx/runtime/include/etx" "primitives.h" md5 0xfb32d66fe9cf38506def6688fb3718fb
	.loc	1 30 1 prologue_end             ; etx/runtime/include/etx/primitives.h:30:1 @[ /tmp/etx-verifygaps/polling/poll.hip:8:3 ]
	s_mov_b32 s6, 1
	s_waitcnt lgkmcnt(0)
	global_load_dword v1, v0, s[2:3] sc1
	s_waitcnt vmcnt(0)
	v_cmp_gt_i32_e32 vcc, 1, v1
	s_cbranch_vccnz .LBB0_7
; %bb.1:                                ; %.preheader
	.loc	1 0 1 is_stmt 0                 ; etx/runtime/include/etx/primitives.h:0:1
	s_load_dwordx2 s[0:1], s[0:1], 0xe8
	s_branch .LBB0_3
.LBB0_2:                                ; %Flow3
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_and_b64 s[4:5], s[4:5], exec
	s_cselect_b32 s4, 1, 0
	s_cmp_lg_u32 s4, 1
	s_cbranch_scc0 .LBB0_7
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 30 1 is_stmt 1                ; etx/runtime/include/etx/primitives.h:30:1 @[ /tmp/etx-verifygaps/polling/poll.hip:8:3 ]
	s_and_b32 s4, s6, 0xff
	s_cselect_b64 s[4:5], -1, 0
	s_and_b64 vcc, exec, s[4:5]
	s_sleep 8
	s_cbranch_vccnz .LBB0_5
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	s_waitcnt lgkmcnt(0)
	global_load_dword v1, v0, s[0:1] sc1
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s4, v1
	s_cmp_eq_u32 s4, 0
	s_cselect_b64 s[4:5], -1, 0
.LBB0_5:                                ; %Flow
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 1 is_stmt 0                 ; etx/runtime/include/etx/primitives.h:0:1
	s_and_b64 s[4:5], s[4:5], exec
	s_cselect_b32 s4, 1, 0
	s_cmp_lg_u32 s4, 1
	s_mov_b64 s[4:5], -1
	s_cbranch_scc1 .LBB0_2
; %bb.6:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 30 1 is_stmt 1                ; etx/runtime/include/etx/primitives.h:30:1 @[ /tmp/etx-verifygaps/polling/poll.hip:8:3 ]
	global_load_dword v1, v0, s[2:3] sc1
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s4, v1
	s_cmp_lt_i32 s4, 1
	s_cselect_b64 s[4:5], -1, 0
	s_add_i32 s6, s6, 1
	s_branch .LBB0_2
.Ltmp1:
.LBB0_7:
	.loc	0 9 1                           ; /tmp/etx-verifygaps/polling/poll.hip:9:1
	s_endpgm
.Ltmp2:
.Lfunc_end0:
	.size	k_polling_wave0, .Lfunc_end0-k_polling_wave0
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_polling_wave0
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 312
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 7
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .Lk_polling_wave0.num_vgpr, 2
	.set .Lk_polling_wave0.num_agpr, 0
	.set .Lk_polling_wave0.numbered_sgpr, 7
	.set .Lk_polling_wave0.num_named_barrier, 0
	.set .Lk_polling_wave0.private_seg_size, 0
	.set .Lk_polling_wave0.uses_vcc, 1
	.set .Lk_polling_wave0.uses_flat_scratch, 0
	.set .Lk_polling_wave0.has_dyn_sized_stack, 0
	.set .Lk_polling_wave0.has_recursion, 0
	.set .Lk_polling_wave0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 176
; TotalNumSgprs: 13
; NumVgprs: 2
; NumAgprs: 0
; TotalNumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 13
; NumVGPRsForWavesPerEU: 2
; AccumOffset: 4
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 0
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_3dbbf2848b4b0d80,@object ; @__hip_cuid_3dbbf2848b4b0d80
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_3dbbf2848b4b0d80
__hip_cuid_3dbbf2848b4b0d80:
	.byte	0                               ; 0x0
	.size	__hip_cuid_3dbbf2848b4b0d80, 1

	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	37                              ; DW_FORM_strx1
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	114                             ; DW_AT_str_offsets_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	37                              ; DW_FORM_strx1
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	115                             ; DW_AT_addr_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	32                              ; DW_AT_inline
	.byte	33                              ; DW_FORM_implicit_const
	.byte	1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	122                             ; DW_AT_call_all_calls
	.byte	25                              ; DW_FORM_flag_present
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	5                               ; DWARF version number
	.byte	1                               ; DWARF Unit Type
	.byte	8                               ; Address Size (in bytes)
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	1                               ; Abbrev [1] 0xc:0x2f DW_TAG_compile_unit
	.byte	0                               ; DW_AT_producer
	.short	48                              ; DW_AT_language
	.byte	1                               ; DW_AT_name
	.long	.Lstr_offsets_base0             ; DW_AT_str_offsets_base
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.byte	2                               ; DW_AT_comp_dir
	.byte	0                               ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	.Laddr_table_base0              ; DW_AT_addr_base
	.byte	2                               ; Abbrev [2] 0x23:0x2 DW_TAG_subprogram
	.byte	3                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x25:0x15 DW_TAG_subprogram
	.byte	0                               ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	4                               ; DW_AT_name
	.byte	4                               ; Abbrev [4] 0x2c:0xd DW_TAG_inlined_subroutine
	.long	35                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_low_pc
	.long	.Ltmp1-.Ltmp0                   ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.byte	8                               ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	24                              ; Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"Homebrew clang version 23.1.1" ; string offset=0 ; Homebrew clang version 23.1.1
.Linfo_string1:
	.asciz	"/tmp/etx-verifygaps/polling/poll.hip" ; string offset=30 ; /tmp/etx-verifygaps/polling/poll.hip
.Linfo_string2:
	.asciz	"/Users/ckl/code/.wt/etx-verifygaps" ; string offset=67 ; /Users/ckl/code/.wt/etx-verifygaps
.Linfo_string3:
	.asciz	"etx_wait_DEVICE"               ; string offset=102 ; etx_wait_DEVICE
.Linfo_string4:
	.asciz	"k_polling_wave0"               ; string offset=118 ; k_polling_wave0
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 ; Length of contribution
.Ldebug_addr_start0:
	.short	5                               ; DWARF version number
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Ltmp0
.Ldebug_addr_end0:
	.ident	"Homebrew clang version 23.1.1"
	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           304
        .value_kind:     by_value
      - .address_space:  global
        .offset:         304
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 312
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           k_polling_wave0
    .private_segment_fixed_size: 0
    .sgpr_count:     13
    .sgpr_spill_count: 0
    .symbol:         k_polling_wave0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     2
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
