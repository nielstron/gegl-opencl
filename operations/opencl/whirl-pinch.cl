/* This file is an image processing operation for GEGL
 *
 * GEGL is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * GEGL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GEGL; if not, see <https://www.gnu.org/licenses/>.
 */

/* gegl:whirl-pinch resamples its input with GeglSamplerNohalo.  The
 * sampler code below is a float4 port of gegl/buffer/gegl-sampler-nohalo.c
 * (operating on "RaGaBaA float" pixels), and fetch () reproduces the way
 * GeglSampler reads pixels around the input's abyss rectangle.
 */

/* Abyss policies, matching GeglAbyssPolicy */
#define ABYSS_NONE  0
#define ABYSS_CLAMP 1
#define ABYSS_LOOP  2
#define ABYSS_BLACK 3
#define ABYSS_WHITE 4

/* Fetch one pixel of the input like GeglSampler does: coordinates
 * outside the buffer's abyss rectangle are handled according to the
 * abyss policy, everything else is read from the uploaded source
 * rectangle 'srect' (x, y, width, height).
 */
static inline float4
fetch (__global const float4 *src,
       int4                   srect,
       int4                   abyss,
       int                    policy,
       int                    x,
       int                    y)
{
  if (x < abyss.x || y < abyss.y ||
      x >= abyss.x + abyss.z || y >= abyss.y + abyss.w)
    {
      switch (policy)
        {
        case ABYSS_CLAMP:
          x = clamp (x, abyss.x, abyss.x + abyss.z - 1);
          y = clamp (y, abyss.y, abyss.y + abyss.w - 1);
          break;
        case ABYSS_LOOP:
          x = abyss.x + (((x - abyss.x) % abyss.z) + abyss.z) % abyss.z;
          y = abyss.y + (((y - abyss.y) % abyss.w) + abyss.w) % abyss.w;
          break;
        case ABYSS_BLACK:
          return (float4) (0.0f, 0.0f, 0.0f, 1.0f);
        case ABYSS_WHITE:
          return (float4) (1.0f);
        default:
          return (float4) (0.0f);
        }
    }

  x = clamp (x - srect.x, 0, srect.z - 1);
  y = clamp (y - srect.y, 0, srect.w - 1);

  return src[y * srect.z + x];
}

#define FETCH(x, y) fetch (src, srect, abyss, policy, (x), (y))
#define NOHALO_MINMOD(a,b,aa,ab) select ((float4) 0.0f, select (b, a, (aa) <= (ab)), (ab) >= 0.0f)
#define NOHALO_MIN(x,y)  select (y, x, (x) <= (y))
#define NOHALO_MAX(x,y)  select (y, x, (x) >= (y))
#define NOHALO_ABS(x)    fabs (x)
#define NOHALO_SIGN(x)   select ((float4) -1.0f, (float4) 1.0f, (x) >= 0.0f)
static inline void
nohalo_subdivision (const float4 uno_two,
                    const float4 uno_thr,
                    const float4 uno_fou,
                    const float4 dos_one,
                    const float4 dos_two,
                    const float4 dos_thr,
                    const float4 dos_fou,
                    const float4 dos_fiv,
                    const float4 tre_one,
                    const float4 tre_two,
                    const float4 tre_thr,
                    const float4 tre_fou,
                    const float4 tre_fiv,
                    const float4 qua_one,
                    const float4 qua_two,
                    const float4 qua_thr,
                    const float4 qua_fou,
                    const float4 qua_fiv,
                    const float4 cin_two,
                    const float4 cin_thr,
                    const float4 cin_fou,
                          float4 *uno_one_1,
                          float4 *uno_two_1,
                          float4 *uno_thr_1,
                          float4 *uno_fou_1,
                          float4 *dos_one_1,
                          float4 *dos_two_1,
                          float4 *dos_thr_1,
                          float4 *dos_fou_1,
                          float4 *tre_one_1,
                          float4 *tre_two_1,
                          float4 *tre_thr_1,
                          float4 *tre_fou_1,
                          float4 *qua_one_1,
                          float4 *qua_two_1,
                          float4 *qua_thr_1,
                          float4 *qua_fou_1)
{
  const float4 d_unodos_two = dos_two - uno_two;
  const float4 d_dostre_two = tre_two - dos_two;
  const float4 d_trequa_two = qua_two - tre_two;
  const float4 d_quacin_two = cin_two - qua_two;
  const float4 d_unodos_thr = dos_thr - uno_thr;
  const float4 d_dostre_thr = tre_thr - dos_thr;
  const float4 d_trequa_thr = qua_thr - tre_thr;
  const float4 d_quacin_thr = cin_thr - qua_thr;
  const float4 d_unodos_fou = dos_fou - uno_fou;
  const float4 d_dostre_fou = tre_fou - dos_fou;
  const float4 d_trequa_fou = qua_fou - tre_fou;
  const float4 d_quacin_fou = cin_fou - qua_fou;
  const float4 d_dos_onetwo = dos_two - dos_one;
  const float4 d_dos_twothr = dos_thr - dos_two;
  const float4 d_dos_thrfou = dos_fou - dos_thr;
  const float4 d_dos_foufiv = dos_fiv - dos_fou;
  const float4 d_tre_onetwo = tre_two - tre_one;
  const float4 d_tre_twothr = tre_thr - tre_two;
  const float4 d_tre_thrfou = tre_fou - tre_thr;
  const float4 d_tre_foufiv = tre_fiv - tre_fou;
  const float4 d_qua_onetwo = qua_two - qua_one;
  const float4 d_qua_twothr = qua_thr - qua_two;
  const float4 d_qua_thrfou = qua_fou - qua_thr;
  const float4 d_qua_foufiv = qua_fiv - qua_fou;
  const float4 d_unodos_times_dostre_two = d_unodos_two * d_dostre_two;
  const float4 d_dostre_two_sq           = d_dostre_two * d_dostre_two;
  const float4 d_dostre_times_trequa_two = d_dostre_two * d_trequa_two;
  const float4 d_trequa_times_quacin_two = d_quacin_two * d_trequa_two;
  const float4 d_quacin_two_sq           = d_quacin_two * d_quacin_two;
  const float4 d_unodos_times_dostre_thr = d_unodos_thr * d_dostre_thr;
  const float4 d_dostre_thr_sq           = d_dostre_thr * d_dostre_thr;
  const float4 d_dostre_times_trequa_thr = d_trequa_thr * d_dostre_thr;
  const float4 d_trequa_times_quacin_thr = d_trequa_thr * d_quacin_thr;
  const float4 d_quacin_thr_sq           = d_quacin_thr * d_quacin_thr;
  const float4 d_unodos_times_dostre_fou = d_unodos_fou * d_dostre_fou;
  const float4 d_dostre_fou_sq           = d_dostre_fou * d_dostre_fou;
  const float4 d_dostre_times_trequa_fou = d_trequa_fou * d_dostre_fou;
  const float4 d_trequa_times_quacin_fou = d_trequa_fou * d_quacin_fou;
  const float4 d_quacin_fou_sq           = d_quacin_fou * d_quacin_fou;
  const float4 d_dos_onetwo_times_twothr = d_dos_onetwo * d_dos_twothr;
  const float4 d_dos_twothr_sq           = d_dos_twothr * d_dos_twothr;
  const float4 d_dos_twothr_times_thrfou = d_dos_twothr * d_dos_thrfou;
  const float4 d_dos_thrfou_times_foufiv = d_dos_thrfou * d_dos_foufiv;
  const float4 d_dos_foufiv_sq           = d_dos_foufiv * d_dos_foufiv;
  const float4 d_tre_onetwo_times_twothr = d_tre_onetwo * d_tre_twothr;
  const float4 d_tre_twothr_sq           = d_tre_twothr * d_tre_twothr;
  const float4 d_tre_twothr_times_thrfou = d_tre_thrfou * d_tre_twothr;
  const float4 d_tre_thrfou_times_foufiv = d_tre_thrfou * d_tre_foufiv;
  const float4 d_tre_foufiv_sq           = d_tre_foufiv * d_tre_foufiv;
  const float4 d_qua_onetwo_times_twothr = d_qua_onetwo * d_qua_twothr;
  const float4 d_qua_twothr_sq           = d_qua_twothr * d_qua_twothr;
  const float4 d_qua_twothr_times_thrfou = d_qua_thrfou * d_qua_twothr;
  const float4 d_qua_thrfou_times_foufiv = d_qua_thrfou * d_qua_foufiv;
  const float4 d_qua_foufiv_sq           = d_qua_foufiv * d_qua_foufiv;
  const float4 dos_thr_y = NOHALO_MINMOD( d_dostre_thr, d_unodos_thr,
                                          d_dostre_thr_sq,
                                          d_unodos_times_dostre_thr );
  const float4 tre_thr_y = NOHALO_MINMOD( d_dostre_thr, d_trequa_thr,
                                          d_dostre_thr_sq,
                                          d_dostre_times_trequa_thr );
  const float4 newval_uno_two =
    0.5f
    *
    ( dos_thr + tre_thr + 0.5f * ( dos_thr_y - tre_thr_y ) );
  const float4 qua_thr_y = NOHALO_MINMOD( d_quacin_thr, d_trequa_thr,
                                          d_quacin_thr_sq,
                                          d_trequa_times_quacin_thr );
  const float4 newval_tre_two =
    0.5f
    *
    ( tre_thr + qua_thr + 0.5f * ( tre_thr_y - qua_thr_y ) );
  const float4 tre_fou_y = NOHALO_MINMOD( d_dostre_fou, d_trequa_fou,
                                          d_dostre_fou_sq,
                                          d_dostre_times_trequa_fou );
  const float4 qua_fou_y = NOHALO_MINMOD( d_quacin_fou, d_trequa_fou,
                                          d_quacin_fou_sq,
                                          d_trequa_times_quacin_fou );
  const float4 newval_tre_fou =
    0.5f
    *
    ( tre_fou + qua_fou + 0.5f * ( tre_fou_y - qua_fou_y ) );
  const float4 dos_fou_y = NOHALO_MINMOD( d_dostre_fou, d_unodos_fou,
                                          d_dostre_fou_sq,
                                          d_unodos_times_dostre_fou );
  const float4 newval_uno_fou =
    0.5f
    *
    ( dos_fou + tre_fou + 0.5f * (dos_fou_y - tre_fou_y ) );
  const float4 tre_two_x = NOHALO_MINMOD( d_tre_twothr, d_tre_onetwo,
                                          d_tre_twothr_sq,
                                          d_tre_onetwo_times_twothr );
  const float4 tre_thr_x = NOHALO_MINMOD( d_tre_twothr, d_tre_thrfou,
                                          d_tre_twothr_sq,
                                          d_tre_twothr_times_thrfou );
  const float4 newval_dos_one =
    0.5f
    *
    ( tre_two + tre_thr + 0.5f * ( tre_two_x - tre_thr_x ) );
  const float4 tre_fou_x = NOHALO_MINMOD( d_tre_foufiv, d_tre_thrfou,
                                          d_tre_foufiv_sq,
                                          d_tre_thrfou_times_foufiv );
  const float4 tre_thr_x_minus_tre_fou_x =
    tre_thr_x - tre_fou_x;
  const float4 newval_dos_thr =
    0.5f
    *
    ( tre_thr + tre_fou + 0.5f * tre_thr_x_minus_tre_fou_x );
  const float4 qua_thr_x = NOHALO_MINMOD( d_qua_twothr, d_qua_thrfou,
                                          d_qua_twothr_sq,
                                          d_qua_twothr_times_thrfou );
  const float4 qua_fou_x = NOHALO_MINMOD( d_qua_foufiv, d_qua_thrfou,
                                          d_qua_foufiv_sq,
                                          d_qua_thrfou_times_foufiv );
  const float4 qua_thr_x_minus_qua_fou_x =
    qua_thr_x - qua_fou_x;
  const float4 newval_qua_thr =
    0.5f
    *
    ( qua_thr + qua_fou + 0.5f * qua_thr_x_minus_qua_fou_x );
  const float4 qua_two_x = NOHALO_MINMOD( d_qua_twothr, d_qua_onetwo,
                                          d_qua_twothr_sq,
                                          d_qua_onetwo_times_twothr );
  const float4 newval_qua_one =
    0.5f
    *
    ( qua_two + qua_thr + 0.5f * ( qua_two_x - qua_thr_x ) );
  const float4 newval_tre_thr =
    0.5f
    *
    (
      newval_tre_two + newval_tre_fou
      +
      0.25f * ( tre_thr_x_minus_tre_fou_x + qua_thr_x_minus_qua_fou_x )
    );
  const float4 dos_thr_x = NOHALO_MINMOD( d_dos_twothr, d_dos_thrfou,
                                          d_dos_twothr_sq,
                                          d_dos_twothr_times_thrfou );
  const float4 dos_fou_x = NOHALO_MINMOD( d_dos_foufiv, d_dos_thrfou,
                                          d_dos_foufiv_sq,
                                          d_dos_thrfou_times_foufiv );
  const float4 newval_uno_thr =
    0.5f
    *
    (
      newval_uno_two + newval_dos_thr
      +
      0.5f
      *
      (
        dos_fou - tre_thr
        +
        0.5f * ( dos_fou_y - tre_fou_y + dos_thr_x - dos_fou_x )
      )
    );
  const float4 tre_two_y = NOHALO_MINMOD( d_dostre_two, d_trequa_two,
                                          d_dostre_two_sq,
                                          d_dostre_times_trequa_two );
  const float4 qua_two_y = NOHALO_MINMOD( d_quacin_two, d_trequa_two,
                                          d_quacin_two_sq,
                                          d_trequa_times_quacin_two );
  const float4 newval_tre_one =
    0.5f
    *
    (
      newval_dos_one + newval_tre_two
      +
      0.5f
      *
      (
        qua_two - tre_thr
        +
        0.5f * ( qua_two_x - qua_thr_x + tre_two_y - qua_two_y )
      )
    );
  const float4 dos_two_x = NOHALO_MINMOD( d_dos_twothr, d_dos_onetwo,
                                          d_dos_twothr_sq,
                                          d_dos_onetwo_times_twothr );
  const float4 dos_two_y = NOHALO_MINMOD( d_dostre_two, d_unodos_two,
                                          d_dostre_two_sq,
                                          d_unodos_times_dostre_two );
  const float4 newval_uno_one =
    0.25f
    *
    (
      dos_two + dos_thr + tre_two + tre_thr
      +
      0.5f
      *
      (
        dos_two_x - dos_thr_x + tre_two_x - tre_thr_x
        +
        dos_two_y + dos_thr_y - tre_two_y - tre_thr_y
      )
    );
  *uno_one_1 = newval_uno_one;
  *uno_two_1 = newval_uno_two;
  *uno_thr_1 = newval_uno_thr;
  *uno_fou_1 = newval_uno_fou;
  *dos_one_1 = newval_dos_one;
  *dos_two_1 =        tre_thr;
  *dos_thr_1 = newval_dos_thr;
  *dos_fou_1 =        tre_fou;
  *tre_one_1 = newval_tre_one;
  *tre_two_1 = newval_tre_two;
  *tre_thr_1 = newval_tre_thr;
  *tre_fou_1 = newval_tre_fou;
  *qua_one_1 = newval_qua_one;
  *qua_two_1 =        qua_thr;
  *qua_thr_1 = newval_qua_thr;
  *qua_fou_1 =        qua_fou;
}
static inline float4
lbb (const float  c00,
     const float  c10,
     const float  c01,
     const float  c11,
     const float  c00dx,
     const float  c10dx,
     const float  c01dx,
     const float  c11dx,
     const float  c00dy,
     const float  c10dy,
     const float  c01dy,
     const float  c11dy,
     const float  c00dxdy,
     const float  c10dxdy,
     const float  c01dxdy,
     const float  c11dxdy,
     const float4 uno_one,
     const float4 uno_two,
     const float4 uno_thr,
     const float4 uno_fou,
     const float4 dos_one,
     const float4 dos_two,
     const float4 dos_thr,
     const float4 dos_fou,
     const float4 tre_one,
     const float4 tre_two,
     const float4 tre_thr,
     const float4 tre_fou,
     const float4 qua_one,
     const float4 qua_two,
     const float4 qua_thr,
     const float4 qua_fou )
{
  const float4 m1    = NOHALO_MIN(dos_two, dos_thr)  ;
  const float4 M1    = NOHALO_MAX(dos_two, dos_thr)  ;
  const float4 m2    = NOHALO_MIN(tre_two, tre_thr)  ;
  const float4 M2    = NOHALO_MAX(tre_two, tre_thr)  ;
  const float4 m4    = NOHALO_MIN(qua_two, qua_thr)  ;
  const float4 M4    = NOHALO_MAX(qua_two, qua_thr)  ;
  const float4 m3    = NOHALO_MIN(uno_two, uno_thr)  ;
  const float4 M3    = NOHALO_MAX(uno_two, uno_thr)  ;
  const float4 m5    = NOHALO_MIN(            m1,       m2      );
  const float4 M5    = NOHALO_MAX(            M1,       M2      );
  const float4 m6    = NOHALO_MIN(dos_one, tre_one)  ;
  const float4 M6    = NOHALO_MAX(dos_one, tre_one)  ;
  const float4 m7    = NOHALO_MIN(dos_fou, tre_fou)  ;
  const float4 M7    = NOHALO_MAX(dos_fou, tre_fou)  ;
  const float4 m13   = NOHALO_MIN(dos_fou, qua_fou)  ;
  const float4 M13   = NOHALO_MAX(dos_fou, qua_fou)  ;
  const float4 m9    = NOHALO_MIN(            m5,       m4      );
  const float4 M9    = NOHALO_MAX(            M5,       M4      );
  const float4 m11   = NOHALO_MIN(            m6,       qua_one );
  const float4 M11   = NOHALO_MAX(            M6,       qua_one );
  const float4 m10   = NOHALO_MIN(            m6,       uno_one );
  const float4 M10   = NOHALO_MAX(            M6,       uno_one );
  const float4 m8    = NOHALO_MIN(            m5,       m3      );
  const float4 M8    = NOHALO_MAX(            M5,       M3      );
  const float4 m12   = NOHALO_MIN(            m7,       uno_fou );
  const float4 M12   = NOHALO_MAX(            M7,       uno_fou );
  const float4 min11 = NOHALO_MIN(            m9,       m13     );
  const float4 max11 = NOHALO_MAX(            M9,       M13     );
  const float4 min01 = NOHALO_MIN(            m9,       m11     );
  const float4 max01 = NOHALO_MAX(            M9,       M11     );
  const float4 min00 = NOHALO_MIN(            m8,       m10     );
  const float4 max00 = NOHALO_MAX(            M8,       M10     );
  const float4 min10 = NOHALO_MIN(            m8,       m12     );
  const float4 max10 = NOHALO_MAX(            M8,       M12     );
  const float4 u11 = tre_thr - min11;
  const float4 v11 = max11 - tre_thr;
  const float4 u01 = tre_two - min01;
  const float4 v01 = max01 - tre_two;
  const float4 u00 = dos_two - min00;
  const float4 v00 = max00 - dos_two;
  const float4 u10 = dos_thr - min10;
  const float4 v10 = max10 - dos_thr;
  const float4 dble_dzdx00i = dos_thr - dos_one;
  const float4 dble_dzdy11i = qua_thr - dos_thr;
  const float4 dble_dzdx10i = dos_fou - dos_two;
  const float4 dble_dzdy01i = qua_two - dos_two;
  const float4 dble_dzdx01i = tre_thr - tre_one;
  const float4 dble_dzdy10i = tre_thr - uno_thr;
  const float4 dble_dzdx11i = tre_fou - tre_two;
  const float4 dble_dzdy00i = tre_two - uno_two;
  const float4 sign_dzdx00 = NOHALO_SIGN( dble_dzdx00i );
  const float4 sign_dzdx10 = NOHALO_SIGN( dble_dzdx10i );
  const float4 sign_dzdx01 = NOHALO_SIGN( dble_dzdx01i );
  const float4 sign_dzdx11 = NOHALO_SIGN( dble_dzdx11i );
  const float4 sign_dzdy00 = NOHALO_SIGN( dble_dzdy00i );
  const float4 sign_dzdy10 = NOHALO_SIGN( dble_dzdy10i );
  const float4 sign_dzdy01 = NOHALO_SIGN( dble_dzdy01i );
  const float4 sign_dzdy11 = NOHALO_SIGN( dble_dzdy11i );
  const float4 quad_d2zdxdy00i = uno_one - uno_thr + dble_dzdx01i;
  const float4 quad_d2zdxdy10i = uno_two - uno_fou + dble_dzdx11i;
  const float4 quad_d2zdxdy01i = qua_thr - qua_one - dble_dzdx00i;
  const float4 quad_d2zdxdy11i = qua_fou - qua_two - dble_dzdx10i;
  const float4 dble_slopelimit_00 = 6.0f * NOHALO_MIN( u00, v00 );
  const float4 dble_slopelimit_10 = 6.0f * NOHALO_MIN( u10, v10 );
  const float4 dble_slopelimit_01 = 6.0f * NOHALO_MIN( u01, v01 );
  const float4 dble_slopelimit_11 = 6.0f * NOHALO_MIN( u11, v11 );
  const float4 dble_dzdx00 =
    select (sign_dzdx00 * dble_slopelimit_00, dble_dzdx00i, sign_dzdx00 * dble_dzdx00i <= dble_slopelimit_00);
  const float4 dble_dzdy00 =
    select (sign_dzdy00 * dble_slopelimit_00, dble_dzdy00i, sign_dzdy00 * dble_dzdy00i <= dble_slopelimit_00);
  const float4 dble_dzdx10 =
    select (sign_dzdx10 * dble_slopelimit_10, dble_dzdx10i, sign_dzdx10 * dble_dzdx10i <= dble_slopelimit_10);
  const float4 dble_dzdy10 =
    select (sign_dzdy10 * dble_slopelimit_10, dble_dzdy10i, sign_dzdy10 * dble_dzdy10i <= dble_slopelimit_10);
  const float4 dble_dzdx01 =
    select (sign_dzdx01 * dble_slopelimit_01, dble_dzdx01i, sign_dzdx01 * dble_dzdx01i <= dble_slopelimit_01);
  const float4 dble_dzdy01 =
    select (sign_dzdy01 * dble_slopelimit_01, dble_dzdy01i, sign_dzdy01 * dble_dzdy01i <= dble_slopelimit_01);
  const float4 dble_dzdx11 =
    select (sign_dzdx11 * dble_slopelimit_11, dble_dzdx11i, sign_dzdx11 * dble_dzdx11i <= dble_slopelimit_11);
  const float4 dble_dzdy11 =
    select (sign_dzdy11 * dble_slopelimit_11, dble_dzdy11i, sign_dzdy11 * dble_dzdy11i <= dble_slopelimit_11);
  const float4 twelve_sum00 = 6.0f * ( dble_dzdx00 + dble_dzdy00 );
  const float4 twelve_dif00 = 6.0f * ( dble_dzdx00 - dble_dzdy00 );
  const float4 twelve_sum10 = 6.0f * ( dble_dzdx10 + dble_dzdy10 );
  const float4 twelve_dif10 = 6.0f * ( dble_dzdx10 - dble_dzdy10 );
  const float4 twelve_sum01 = 6.0f * ( dble_dzdx01 + dble_dzdy01 );
  const float4 twelve_dif01 = 6.0f * ( dble_dzdx01 - dble_dzdy01 );
  const float4 twelve_sum11 = 6.0f * ( dble_dzdx11 + dble_dzdy11 );
  const float4 twelve_dif11 = 6.0f * ( dble_dzdx11 - dble_dzdy11 );
  const float4 twelve_abs_sum00 = NOHALO_ABS( twelve_sum00 );
  const float4 twelve_abs_sum10 = NOHALO_ABS( twelve_sum10 );
  const float4 twelve_abs_sum01 = NOHALO_ABS( twelve_sum01 );
  const float4 twelve_abs_sum11 = NOHALO_ABS( twelve_sum11 );
  const float4 u00_times_36 = 36.0f * u00;
  const float4 u10_times_36 = 36.0f * u10;
  const float4 u01_times_36 = 36.0f * u01;
  const float4 u11_times_36 = 36.0f * u11;
  const float4 first_limit00 = twelve_abs_sum00 - u00_times_36;
  const float4 first_limit10 = twelve_abs_sum10 - u10_times_36;
  const float4 first_limit01 = twelve_abs_sum01 - u01_times_36;
  const float4 first_limit11 = twelve_abs_sum11 - u11_times_36;
  const float4 quad_d2zdxdy00ii = NOHALO_MAX( quad_d2zdxdy00i, first_limit00 );
  const float4 quad_d2zdxdy10ii = NOHALO_MAX( quad_d2zdxdy10i, first_limit10 );
  const float4 quad_d2zdxdy01ii = NOHALO_MAX( quad_d2zdxdy01i, first_limit01 );
  const float4 quad_d2zdxdy11ii = NOHALO_MAX( quad_d2zdxdy11i, first_limit11 );
  const float4 v00_times_36 = 36.0f * v00;
  const float4 v10_times_36 = 36.0f * v10;
  const float4 v01_times_36 = 36.0f * v01;
  const float4 v11_times_36 = 36.0f * v11;
  const float4 second_limit00 = v00_times_36 - twelve_abs_sum00;
  const float4 second_limit10 = v10_times_36 - twelve_abs_sum10;
  const float4 second_limit01 = v01_times_36 - twelve_abs_sum01;
  const float4 second_limit11 = v11_times_36 - twelve_abs_sum11;
  const float4 quad_d2zdxdy00iii =
    NOHALO_MIN( quad_d2zdxdy00ii, second_limit00 );
  const float4 quad_d2zdxdy10iii =
    NOHALO_MIN( quad_d2zdxdy10ii, second_limit10 );
  const float4 quad_d2zdxdy01iii =
    NOHALO_MIN( quad_d2zdxdy01ii, second_limit01 );
  const float4 quad_d2zdxdy11iii =
    NOHALO_MIN( quad_d2zdxdy11ii, second_limit11 );
  const float4 twelve_abs_dif00 = NOHALO_ABS( twelve_dif00 );
  const float4 twelve_abs_dif10 = NOHALO_ABS( twelve_dif10 );
  const float4 twelve_abs_dif01 = NOHALO_ABS( twelve_dif01 );
  const float4 twelve_abs_dif11 = NOHALO_ABS( twelve_dif11 );
  const float4 third_limit00 = twelve_abs_dif00 - v00_times_36;
  const float4 third_limit10 = twelve_abs_dif10 - v10_times_36;
  const float4 third_limit01 = twelve_abs_dif01 - v01_times_36;
  const float4 third_limit11 = twelve_abs_dif11 - v11_times_36;
  const float4 quad_d2zdxdy00iiii =
    NOHALO_MAX( quad_d2zdxdy00iii, third_limit00);
  const float4 quad_d2zdxdy10iiii =
    NOHALO_MAX( quad_d2zdxdy10iii, third_limit10);
  const float4 quad_d2zdxdy01iiii =
    NOHALO_MAX( quad_d2zdxdy01iii, third_limit01);
  const float4 quad_d2zdxdy11iiii =
    NOHALO_MAX( quad_d2zdxdy11iii, third_limit11);
  const float4 fourth_limit00 = u00_times_36 - twelve_abs_dif00;
  const float4 fourth_limit10 = u10_times_36 - twelve_abs_dif10;
  const float4 fourth_limit01 = u01_times_36 - twelve_abs_dif01;
  const float4 fourth_limit11 = u11_times_36 - twelve_abs_dif11;
  const float4 quad_d2zdxdy00 = NOHALO_MIN( quad_d2zdxdy00iiii, fourth_limit00);
  const float4 quad_d2zdxdy10 = NOHALO_MIN( quad_d2zdxdy10iiii, fourth_limit10);
  const float4 quad_d2zdxdy01 = NOHALO_MIN( quad_d2zdxdy01iiii, fourth_limit01);
  const float4 quad_d2zdxdy11 = NOHALO_MIN( quad_d2zdxdy11iiii, fourth_limit11);
  const float4 newval1 = c00 * dos_two
                         +
                         c10 * dos_thr
                         +
                         c01 * tre_two
                         +
                         c11 * tre_thr;
  const float4 newval2 = c00dx * dble_dzdx00
                         +
                         c10dx * dble_dzdx10
                         +
                         c01dx * dble_dzdx01
                         +
                         c11dx * dble_dzdx11
                         +
                         c00dy * dble_dzdy00
                         +
                         c10dy * dble_dzdy10
                         +
                         c01dy * dble_dzdy01
                         +
                         c11dy * dble_dzdy11;
  const float4 newval3 = c00dxdy * quad_d2zdxdy00
                         +
                         c10dxdy * quad_d2zdxdy10
                         +
                         c01dxdy * quad_d2zdxdy01
                         +
                         c11dxdy * quad_d2zdxdy11;
  const float4 newval =
    newval1 + 0.5f * ( newval2 + 0.5f * newval3 );
  return newval;
}

static inline float
nohalo_teepee (const float c_major_x,
               const float c_major_y,
               const float c_minor_x,
               const float c_minor_y,
               const float s,
               const float t)
{
  const float q1 = s * c_major_x + t * c_major_y;
  const float q2 = s * c_minor_x + t * c_minor_y;
  const float r2 = q1 * q1 + q2 * q2;

  return r2 < 1.0f ? 1.0f - sqrt (r2) : 0.0f;
}

/* Port of gegl_sampler_nohalo_get (gegl/buffer/gegl-sampler-nohalo.c),
 * computing all four channels at once.  The host has done the parts
 * that depend on the sampling position and the Jacobian only (see
 * wp_pixel_geometry () in whirl-pinch.c): (ix_0, iy_0) is the anchor
 * pixel and (x_0, y_0) the position relative to it; if bounds is not
 * 0, it holds the EWA box (each offset + 64 in one byte) and theta and
 * c_major/c_minor are the EWA blend factor and axes.
 */
static inline float4
nohalo_get (__global const float4 *src,
            int4                   srect,
            int4                   abyss,
            int                    policy,
            const int              ix_0,
            const int              iy_0,
            const float            x_0,
            const float            y_0,
            const int              bounds,
            const float            theta,
            const float4           c_axes)
{
  const int sx = x_0 >= 0.0f ? 1 : -1;
  const int sy = y_0 >= 0.0f ? 1 : -1;

  float4 newval;

  {
    const float xp1over2   = (2 * sx) * x_0;
    const float xm1over2   = xp1over2 + (-1.0f);
    const float onepx      = 0.5f + xp1over2;
    const float onemx      = 1.5f - xp1over2;
    const float xp1over2sq = xp1over2 * xp1over2;

    const float yp1over2   = (2 * sy) * y_0;
    const float ym1over2   = yp1over2 + (-1.0f);
    const float onepy      = 0.5f + yp1over2;
    const float onemy      = 1.5f - yp1over2;
    const float yp1over2sq = yp1over2 * yp1over2;

    const float xm1over2sq = xm1over2 * xm1over2;
    const float ym1over2sq = ym1over2 * ym1over2;

    const float twice1px = onepx + onepx;
    const float twice1py = onepy + onepy;
    const float twice1mx = onemx + onemx;
    const float twice1my = onemy + onemy;

    const float xm1over2sq_times_ym1over2sq = xm1over2sq * ym1over2sq;
    const float xp1over2sq_times_ym1over2sq = xp1over2sq * ym1over2sq;
    const float xp1over2sq_times_yp1over2sq = xp1over2sq * yp1over2sq;
    const float xm1over2sq_times_yp1over2sq = xm1over2sq * yp1over2sq;

    const float four_times_1px_times_1py = twice1px * twice1py;
    const float four_times_1mx_times_1py = twice1mx * twice1py;
    const float twice_xp1over2_times_1py = xp1over2 * twice1py;
    const float twice_xm1over2_times_1py = xm1over2 * twice1py;

    const float twice_xm1over2_times_1my = xm1over2 * twice1my;
    const float twice_xp1over2_times_1my = xp1over2 * twice1my;
    const float four_times_1mx_times_1my = twice1mx * twice1my;
    const float four_times_1px_times_1my = twice1px * twice1my;

    const float twice_1px_times_ym1over2 = twice1px * ym1over2;
    const float twice_1mx_times_ym1over2 = twice1mx * ym1over2;
    const float xp1over2_times_ym1over2  = xp1over2 * ym1over2;
    const float xm1over2_times_ym1over2  = xm1over2 * ym1over2;

    const float xm1over2_times_yp1over2  = xm1over2 * yp1over2;
    const float xp1over2_times_yp1over2  = xp1over2 * yp1over2;
    const float twice_1mx_times_yp1over2 = twice1mx * yp1over2;
    const float twice_1px_times_yp1over2 = twice1px * yp1over2;

    const float c00     = four_times_1px_times_1py * xm1over2sq_times_ym1over2sq;
    const float c00dx   = twice_xp1over2_times_1py * xm1over2sq_times_ym1over2sq;
    const float c00dy   = twice_1px_times_yp1over2 * xm1over2sq_times_ym1over2sq;
    const float c00dxdy =  xp1over2_times_yp1over2 * xm1over2sq_times_ym1over2sq;

    const float c10     = four_times_1mx_times_1py * xp1over2sq_times_ym1over2sq;
    const float c10dx   = twice_xm1over2_times_1py * xp1over2sq_times_ym1over2sq;
    const float c10dy   = twice_1mx_times_yp1over2 * xp1over2sq_times_ym1over2sq;
    const float c10dxdy =  xm1over2_times_yp1over2 * xp1over2sq_times_ym1over2sq;

    const float c01     = four_times_1px_times_1my * xm1over2sq_times_yp1over2sq;
    const float c01dx   = twice_xp1over2_times_1my * xm1over2sq_times_yp1over2sq;
    const float c01dy   = twice_1px_times_ym1over2 * xm1over2sq_times_yp1over2sq;
    const float c01dxdy =  xp1over2_times_ym1over2 * xm1over2sq_times_yp1over2sq;

    const float c11     = four_times_1mx_times_1my * xp1over2sq_times_yp1over2sq;
    const float c11dx   = twice_xm1over2_times_1my * xp1over2sq_times_yp1over2sq;
    const float c11dy   = twice_1mx_times_ym1over2 * xp1over2sq_times_yp1over2sq;
    const float c11dxdy =  xm1over2_times_ym1over2 * xp1over2sq_times_yp1over2sq;

    float4 uno_one, uno_two, uno_thr, uno_fou;
    float4 dos_one, dos_two, dos_thr, dos_fou;
    float4 tre_one, tre_two, tre_thr, tre_fou;
    float4 qua_one, qua_two, qua_thr, qua_fou;

    nohalo_subdivision (FETCH (ix_0 -     sx, iy_0 - 2 * sy),
                        FETCH (ix_0         , iy_0 - 2 * sy),
                        FETCH (ix_0 +     sx, iy_0 - 2 * sy),
                        FETCH (ix_0 - 2 * sx, iy_0 -     sy),
                        FETCH (ix_0 -     sx, iy_0 -     sy),
                        FETCH (ix_0         , iy_0 -     sy),
                        FETCH (ix_0 +     sx, iy_0 -     sy),
                        FETCH (ix_0 + 2 * sx, iy_0 -     sy),
                        FETCH (ix_0 - 2 * sx, iy_0         ),
                        FETCH (ix_0 -     sx, iy_0         ),
                        FETCH (ix_0         , iy_0         ),
                        FETCH (ix_0 +     sx, iy_0         ),
                        FETCH (ix_0 + 2 * sx, iy_0         ),
                        FETCH (ix_0 - 2 * sx, iy_0 +     sy),
                        FETCH (ix_0 -     sx, iy_0 +     sy),
                        FETCH (ix_0         , iy_0 +     sy),
                        FETCH (ix_0 +     sx, iy_0 +     sy),
                        FETCH (ix_0 + 2 * sx, iy_0 +     sy),
                        FETCH (ix_0 -     sx, iy_0 + 2 * sy),
                        FETCH (ix_0         , iy_0 + 2 * sy),
                        FETCH (ix_0 +     sx, iy_0 + 2 * sy),
                        &uno_one, &uno_two, &uno_thr, &uno_fou,
                        &dos_one, &dos_two, &dos_thr, &dos_fou,
                        &tre_one, &tre_two, &tre_thr, &tre_fou,
                        &qua_one, &qua_two, &qua_thr, &qua_fou);

    newval = lbb (c00, c10, c01, c11,
                  c00dx, c10dx, c01dx, c11dx,
                  c00dy, c10dy, c01dy, c11dy,
                  c00dxdy, c10dxdy, c01dxdy, c11dxdy,
                  uno_one, uno_two, uno_thr, uno_fou,
                  dos_one, dos_two, dos_thr, dos_fou,
                  tre_one, tre_two, tre_thr, tre_fou,
                  qua_one, qua_two, qua_thr, qua_fou);
  }

  if (bounds != 0)
    {
      /* EWA downsampling, blended in where the transformation downsamples */
      const int out_left_0 = ( bounds        & 0xff) - 64;
      const int out_rite_0 = ((bounds >>  8) & 0xff) - 64;
      const int out_top_0  = ((bounds >> 16) & 0xff) - 64;
      const int out_bot_0  = ((bounds >> 24) & 0xff) - 64;

      float  total_weight = 0.0f;
      float4 ewa_newval   = (float4) (0.0f);
      int    i, j;

      /* do-while like the sampler, which visits at least one pixel */
      i = out_top_0;
      do
        {
          j = out_left_0;
          do
            {
              const float weight = nohalo_teepee (c_axes.x, c_axes.y,
                                                  c_axes.z, c_axes.w,
                                                  x_0 - (float) j,
                                                  y_0 - (float) i);

              total_weight += weight;
              ewa_newval   += weight * FETCH (ix_0 + j, iy_0 + i);
            }
          while (++j <= out_rite_0);
        }
      while (++i <= out_bot_0);

      {
        const float beta = (1.0f - theta) / total_weight;

        newval = theta * newval + beta * ewa_newval;
      }
    }

  return newval;
}

__kernel void
whirl_pinch (__global const float4 *src,
             int4                   srect,
             int4                   abyss,
             int                    policy,
             __global const int    *geo,
             __global       float4 *dst,
             int                    dst_offset)
{
  const int idx = get_global_id (1) * get_global_size (0) + get_global_id (0);
  __global const int *g = geo + idx * 10;
  const int4 p      = vload4 (0, g);
  const int  bounds = g[4];
  float      theta  = 0.0f;
  float4     c_axes = (float4) (0.0f);

  if (bounds != 0)
    {
      theta  = as_float (g[5]);
      c_axes = as_float4 (vload4 (0, g + 6));
    }

  dst[dst_offset + idx] =
    nohalo_get (src, srect, abyss, policy,
                p.x, p.y, as_float (p.z), as_float (p.w),
                bounds, theta, c_axes);
}
