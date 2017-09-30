;*****************************************************************************
;* x86-optimized functions for vmafmotion filter
;*
;* Copyright (c) 2017 Ronald S. Bultje <rsbultje@gmail.com>
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
cglobal sad, 6, 7, 3, buf1, buf2, w, h, buf1_stride, buf2_stride
    pxor       m0, m0
.loop_y:
    xor          r6, r6
    .loop:
        mova           m1, [buf1q+r6*2]
        mova           m2, [buf2q+r6*2]
        psubw          m1, m2
        pabsw          m1, m1
        pmaddwd        m1, [pw_1]
        paddd          m0, m1
        add            r6, mmsize / 2
        cmp            r6d, wd
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
