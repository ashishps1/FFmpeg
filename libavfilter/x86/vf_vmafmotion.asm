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

%macro ABSW 2 ; img2/img1, tmp
%if cpuflag(ssse3)
    pabsw   %1, %1
%else
    pxor    %2, %2
    psubw   %2, %1
    pmaxsw  %1, %2
%endif
%endmacro

%macro IMAGE_SAD_FN 0
cglobal sad_8x8, 4, 7, 6, img1, img2, img1_stride, img2_stride, \
                          img1_stride3, img2_stride3, cnt
    lea    img1_stride3q, [img1_strideq*3]
    lea    img2_stride3q, [img2_strideq*3]
    mov            cntd, 2
    pxor             m0, m0
.loop:
    mova             m1, [img1q+img1_strideq*0]
    mova             m2, [img1q+img1_strideq*1]
    mova             m3, [img1q+img1_strideq*2]
    mova             m4, [img1q+img1_stride3q]
    lea            img1q, [img1q+img1_strideq*4]
    psubw            m1, [img2q+img2_strideq*0]
    psubw            m2, [img2q+img2_strideq*1]
    psubw            m3, [img2q+img2_strideq*2]
    psubw            m4, [img2q+img2_stride3q]
    lea            img1q, [img2q+img2_strideq*4]
    ABSW             m1, m5
    ABSW             m2, m5
    ABSW             m3, m5
    ABSW             m4, m5
    paddw            m1, m2
    paddw            m3, m4
    paddw            m0, m1
    paddw            m0, m3
    dec            cntd
    jg .loop
    movhlps          m1, m0
    paddw            m0, m1
    pshuflw      m1, m0, q1010
    paddw            m0, m1
    pshuflw      m1, m0, q0000 ; qNNNN is a base4-notation for imm8 arguments
    paddw            m0, m0
    movd            eax, m0
    movsxwd         eax, ax
    RET
%endmacro

INIT_XMM sse2
IMAGE_SAD_FN

INIT_XMM ssse3
IMAGE_SAD_FN
