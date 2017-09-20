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

SECTION .text

INIT_XMM sse3
cglobal sad, 6, 9, 6, buf1, buf2, w, h, buf1_stride, buf2_stride
    pxor       m0, m0
    mov        r0, [hq]
.loop_y:
    mov          r1, 0
    .loop:
        mova           m1, [buf1q]
        mova           m2, [buf2q]
        psubw          m1, m2
        pabsw          m1, m1
        paddw          m0, m1
        add            r1, 1
        cmp            r1, wq
    jl .loop

    add        buf1q, buf1_strideq
    add        buf2q, buf2_strideq
    dec        r0
    jg .loop_y
    RET
