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

/* gegl:polar-coordinates resamples its input with GeglSamplerNohalo.  The
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

#define NOHALO_OFFSET_0 13

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
 * computing all four channels at once.  The 2x2 matrix (a b; c d) is
 * the inverse Jacobian as computed by gegl_sampler_compute_scale.
 */
static inline float4
nohalo_get (__global const float4 *src,
            int4                   srect,
            int4                   abyss,
            int                    policy,
            float                  absolute_x,
            float                  absolute_y,
            float                  a,
            float                  b,
            float                  c,
            float                  d)
{
  const int ix_0 = (int) floor (absolute_x);
  const int iy_0 = (int) floor (absolute_y);

  const float x_0 = (absolute_x - 0.5f) - ix_0;
  const float y_0 = (absolute_y - 0.5f) - iy_0;

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

  {
    /* EWA downsampling, only when the transformation downsamples */
    const float aa  = a * a;
    const float bb  = b * b;
    const float cc  = c * c;
    const float dd  = d * d;
    const float n11 = aa + bb;
    const float n12 = a * c + b * d;
    const float n21 = n12;
    const float n22 = cc + dd;
    const float det = a * d - b * c;
    const float twice_det = det + det;
    const float frobenius_squared = n11 + n22;
    const float discriminant =
      (frobenius_squared + twice_det) * (frobenius_squared - twice_det);
    const float sqrt_discriminant =
      discriminant > 0.0f ? sqrt (discriminant) : 0.0f;
    const float twice_s1s1 = frobenius_squared + sqrt_discriminant;

    if (twice_s1s1 > 2.0f)
      {
        const float s1s1 = 0.5f * twice_s1s1;
        const float s2s2 = 0.5f * (frobenius_squared - sqrt_discriminant);

        const float s1s1minusn11 = s1s1 - n11;
        const float s1s1minusn22 = s1s1 - n22;
        const float s1s1minusn11_squared = s1s1minusn11 * s1s1minusn11;
        const float s1s1minusn22_squared = s1s1minusn22 * s1s1minusn22;
        const float temp_u11 =
          s1s1minusn11_squared >= s1s1minusn22_squared ? n12 : s1s1minusn22;
        const float temp_u21 =
          s1s1minusn11_squared >= s1s1minusn22_squared ? s1s1minusn11 : n21;
        const float norm = sqrt (temp_u11 * temp_u11 + temp_u21 * temp_u21);
        const float u11 = norm > 0.0f ? temp_u11 / norm : 1.0f;
        const float u21 = norm > 0.0f ? temp_u21 / norm : 0.0f;
        const float major_mag = s1s1 <= 1.0f ? 1.0f : sqrt (s1s1);
        const float minor_mag = s2s2 <= 1.0f ? 1.0f : sqrt (s2s2);
        const float major_unit_x =  u11;
        const float major_unit_y =  u21;
        const float minor_unit_x = -u21;
        const float minor_unit_y =  u11;

        const float c_major_x = major_unit_x / major_mag;
        const float c_major_y = major_unit_y / major_mag;
        const float c_minor_x = minor_unit_x / minor_mag;
        const float c_minor_y = minor_unit_y / minor_mag;

        const float major_x = major_mag * major_unit_x;
        const float major_y = major_mag * major_unit_y;
        const float minor_x = minor_mag * minor_unit_x;
        const float minor_y = minor_mag * minor_unit_y;

        const float ellipse_a = major_y * major_y + minor_y * minor_y;
        const float folded_ellipse_b = major_x * major_y + minor_x * minor_y;
        const float ellipse_c = major_x * major_x + minor_x * minor_x;
        const float ellipse_f = major_mag * minor_mag;

        const float bounding_box_factor =
          ellipse_f * ellipse_f /
          (ellipse_c * ellipse_a - folded_ellipse_b * folded_ellipse_b);
        const float bounding_box_half_width =
          sqrt (ellipse_c * bounding_box_factor);
        const float bounding_box_half_height =
          sqrt (ellipse_a * bounding_box_factor);

        const float theta = 1.0f / ellipse_f;

        const int out_left_0 =
          max ((int) ceil (x_0 - bounding_box_half_width), -NOHALO_OFFSET_0);
        const int out_rite_0 =
          min ((int) floor (x_0 + bounding_box_half_width), NOHALO_OFFSET_0);
        const int out_top_0 =
          max ((int) ceil (y_0 - bounding_box_half_height), -NOHALO_OFFSET_0);
        const int out_bot_0 =
          min ((int) floor (y_0 + bounding_box_half_height), NOHALO_OFFSET_0);

        float  total_weight = 0.0f;
        float4 ewa_newval   = (float4) (0.0f);
        int    i, j;

        for (i = out_top_0; i <= out_bot_0; i++)
          for (j = out_left_0; j <= out_rite_0; j++)
            {
              const float weight = nohalo_teepee (c_major_x, c_major_y,
                                                  c_minor_x, c_minor_y,
                                                  x_0 - (float) j,
                                                  y_0 - (float) i);

              total_weight += weight;
              ewa_newval   += weight * FETCH (ix_0 + j, iy_0 + i);
            }

        {
          const float beta = (1.0f - theta) / total_weight;

          newval = theta * newval + beta * ewa_newval;
        }
      }
  }

  return newval;
}

/* "RaGaBaA float" to "RGBA float", like babl does it */
#define ALPHA_FLOOR (1.0f / 65536.0f)

static inline float4
unpremultiply (float4 v)
{
  const float alpha = v.w;
  const float used  = (alpha <= ALPHA_FLOOR && alpha >= -ALPHA_FLOOR) ?
                      ALPHA_FLOOR : alpha;
  const float recip = 1.0f / used;

  return (float4) (v.xyz * recip, alpha);
}

/* GEGL's fmod () (gegl-math.h) is floored, so unlike OpenCL's fmod ()
 * its result has the sign of y, not of x.
 */
static float
gegl_fmod (float x,
           float y)
{
  return x - y * floor (x / y);
}

/* Port of calc_undistorted_coords () in polar-coordinates.c */
static inline int
polar_unmap (float  wx,
             float  wy,
             float  cen_x,
             float  cen_y,
             int    width,
             int    height,
             float  circle,
             float  angl,
             int    bw,
             int    top,
             int    polar,
             float2 col,
             int2   rows,
             float *x,
             float *y)
{
  float phi = 0.0f;
  float r = 0.0f, m;
  float xmax, ymax, rmax, t;
  float x_calc, y_calc;
  const int   x1 = 0, y1 = 0, x2 = width, y2 = height;
  const int   xdiff = x2 - x1, ydiff = y2 - y1;
  const float xm = xdiff / 2.0f, ym = ydiff / 2.0f;
  int xi, yi;

  if (polar)
    {
      if (wx >= cen_x)
        {
          if (wy > cen_y)
            phi = M_PI_F - atan ((wx - cen_x) / (wy - cen_y));
          else if (wy < cen_y)
            phi = atan ((wx - cen_x) / (cen_y - wy));
          else
            phi = M_PI_F / 2;
        }
      else
        {
          if (wy < cen_y)
            phi = 2 * M_PI_F - atan ((cen_x - wx) / (cen_y - wy));
          else if (wy > cen_y)
            phi = M_PI_F + atan ((cen_x - wx) / (wy - cen_y));
          else
            phi = 1.5f * M_PI_F;
        }

      r = sqrt ((wx - cen_x) * (wx - cen_x) + (wy - cen_y) * (wy - cen_y));

      if (wx != cen_x)
        m = fabs ((wy - cen_y) / (wx - cen_x));
      else
        m = 0;

      if (m <= ((float) (y2 - y1) / (float) (x2 - x1)))
        {
          if (wx == cen_x)
            {
              xmax = 0;
              ymax = cen_y - y1;
            }
          else
            {
              xmax = cen_x - x1;
              ymax = m * xmax;
            }
        }
      else
        {
          ymax = cen_y - y1;
          xmax = ymax / m;
        }

      rmax = sqrt (xmax * xmax + ymax * ymax);

      t = ((cen_y - y1) < (cen_x - x1)) ? (cen_y - y1) : (cen_x - x1);
      rmax = (rmax - t) / 100 * (100 - circle) + t;

      phi = gegl_fmod (phi + angl, 2 * M_PI_F);

      if (bw)
        x_calc = x2 - 1 - (x2 - x1 - 1) / (2 * M_PI_F) * phi;
      else
        x_calc = (x2 - x1 - 1) / (2 * M_PI_F) * phi + x1;

      if (top)
        y_calc = (y2 - y1) / rmax * r + y1;
      else
        y_calc = y2 - (y2 - y1) / rmax * r;
    }
  else
    {
      /* col holds the column's +-rmax * sin (phi2) and +-rmax * cos (phi2),
       * computed on the host in double precision
       */
      if (top)
        r = (wy - y1) / (float) ydiff;
      else
        r = (y2 - wy) / (float) ydiff;

      x_calc = xm + col.x * r;
      y_calc = ym + col.y * r;

      /* whether the row is inside is decided on the host in double
       * precision, being too close to the threshold for single precision
       * at times
       */
      *x = x_calc;
      *y = y_calc;

      return rows.x <= (int) wy && (int) wy <= rows.y;
    }

  /* With rmax == 0 (the pole on the top or left edge) the coordinates
   * can be NaN or infinite.  Convert them with saturation, NaN going to
   * 0, as the CPU's (int) cast does on ARM; the sampler then reads
   * non-finite coordinates as 0, like gegl_sampler_get () does.
   */
  xi = convert_int_sat (x_calc + 0.5f);
  yi = convert_int_sat (y_calc + 0.5f);

  *x = isfinite (x_calc) ? x_calc : 0.0f;
  *y = isfinite (y_calc) ? y_calc : 0.0f;

  return 0 <= xi && xi <= width - 1 && 0 <= yi && yi <= height - 1;
}

__kernel void
polar_coordinates (__global const float4 *src,
                   int4                   srect,
                   int4                   abyss,
                   int                    policy,
                   __global       float4 *dst,
                   int                    roi_x,
                   int                    roi_y,
                   float                  cen_x,
                   float                  cen_y,
                   int                    width,
                   int                    height,
                   float                  circle,
                   float                  angl,
                   int                    bw,
                   int                    top,
                   int                    polar,
                   __global const float2 *cols,
                   __global const int2   *rows,
                   int                    cols_x,
                   int                    unpremultiply_output)
{
  const int gidx = get_global_id (0);
  const int gidy = get_global_id (1);
  float4    out  = (float4) (0.0f);
  float     px, py;

  if (polar_unmap (roi_x + gidx, roi_y + gidy, cen_x, cen_y, width, height,
                   circle, angl, bw, top, polar,
                   cols[polar ? 0 : roi_x + gidx - cols_x],
                   rows[polar ? 0 : roi_x + gidx - cols_x], &px, &py))
    {
      /* the CPU code passes an all-zero scale matrix, which disables
       * NoHalo's EWA path */
      out = nohalo_get (src, srect, abyss, policy, px, py,
                        0.0f, 0.0f, 0.0f, 0.0f);

      if (unpremultiply_output)
        out = unpremultiply (out);
    }

  dst[gidy * get_global_size (0) + gidx] = out;
}
