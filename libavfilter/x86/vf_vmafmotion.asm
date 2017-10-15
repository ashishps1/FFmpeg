;*****************************************************************************
;* x86-optimized functions for vmafmotion filter
;*
;* Copyright (c) 2017 Ashish Pratap Singh <ashk43712@gmail.com>
;*
;* This file is part of FFmpeg.
;*
;* FFmpeg is free software; you can redistribute it and/or
;* modify it under the terms of the GNU Lesser General Public
;* License as published by the Free Software Foundation; either
;* version 2.1 of the License, or (at your option) any later version.
;*
;* FFmpeg is distributed in the hope that it will be useful,
;* but WITHOUT ANY WARRANTY; without even the implied warranty of
;* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
;* Lesser General Public License for more details.
;*
;* You should have received a copy of the GNU Lesser General Public
;* License along with FFmpeg; if not, write to the Free Software
;* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
;******************************************************************************

%include "libavutil/x86/x86util.asm"

SECTION_RODATA

pw_1: times 8 dw 1

SECTION .text

INIT_XMM sse3
cglobal sad, 7, 7, 3, buf1, buf2, w, h, buf1_stride, buf2_stride, x
    pxor       m0, m0
.loop_y:
    xor          xq, xq
    .loop:
        mova           m1, [buf1q+xq*2]
        mova           m2, [buf2q+xq*2]
        psubw          m1, m2
        pabsw          m1, m1
        pmaddwd        m1, [pw_1]
        paddd          m0, m1
        add            xq, mmsize / 2
        cmp            xd, wd
    jl .loop

    add        buf1q, buf1_strideq
    add        buf2q, buf2_strideq
    dec        hd
    jg .loop_y
    
    movhlps         m1, m0
    paddd           m0, m1
    pshufd          m1, m0, q0000
    paddd           m0, m1
    movd            eax, m0
    RET
    
cglobal convolution_x, 11, 11, 3, filter, filt_w, src, dst, w, h, src_stride, dst_stride, radius, border_right, temp
    mov        eax, filt_wd
    mov        ecx, 2h
    div        ecx
    mov        radiusd, ecx
    xor        tempq, tempq
    mov        tempq, filt_wq
    sub        tempq, radiusq
    mov        border_rightq, wq
    sub        border_rightq, tempq
    pxor       m2, m2
.loop_h:
    .loop_b1:
        pxor           m0, m0
        .loop_fw1:
	          
	jl .loop_fw1
    jl .loop_b1
    .loop_b2:
        pxor           m0, m0
        .loop_fw2:

	jl .loop_fw2
    jl .loop_b2
    .loop_b3:
        pxor           m0, m0
        .loop_fw3:

	jl .loop_fw3
    jl .loop_b3    
    jl .loop_h
    RET
